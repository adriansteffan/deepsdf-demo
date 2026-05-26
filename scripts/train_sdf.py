import argparse
import csv
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sdf_model import SDFModel


def sdf_loss(pred, gt, latents_in_batch, sigma):
    l1 = (pred - gt).abs().mean()
    l2 = sigma ** 2 * latents_in_batch.norm(dim=1, p=2).mean()
    return (l1 + l2, l1, l2)

def load_dataset(samples_path, clamp_value, device):
    d = np.load(samples_path, allow_pickle=True).item()
    chunks_lc, chunks_co, chunks_sdf = ([], [], [])
    for obj_idx, rec in d.items():
        slc = rec['samples_latent_class']
        sdf = rec['sdf'].reshape(-1)
        chunks_lc.append(slc[:, 0].astype(np.int64))
        chunks_co.append(slc[:, 1:4].astype(np.float32))
        chunks_sdf.append(sdf.astype(np.float32))
    lc = torch.from_numpy(np.concatenate(chunks_lc)).to(device)
    co = torch.from_numpy(np.concatenate(chunks_co)).to(device)
    sd = torch.from_numpy(np.concatenate(chunks_sdf)).to(device)
    sd.clamp_(-clamp_value, clamp_value)
    n_shapes = int(lc.max().item()) + 1
    print(f'loaded {len(sd)} samples across {n_shapes} shapes')
    return (lc, co, sd, n_shapes)

def epoch_indices(n_total, batch_size, generator):
    perm = torch.randperm(n_total, generator=generator, device='cpu')
    n_full = n_total // batch_size * batch_size
    perm = perm[:n_full]
    return perm.reshape(-1, batch_size)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', default='model')
    ap.add_argument('--dataset', default='ShapeNetCore')
    ap.add_argument('--num-layers', type=int, default=8)
    ap.add_argument('--inner-dim', type=int, default=256)
    ap.add_argument('--latent-size', type=int, default=64)
    ap.add_argument('--skip-connections', action='store_true', default=True)
    ap.add_argument('--clamp-value', type=float, default=0.1)
    ap.add_argument('--sigma', type=float, default=0.01)
    ap.add_argument('--batch-size', type=int, default=4096)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--lr-model', type=float, default=0.0001)
    ap.add_argument('--lr-latent', type=float, default=0.001)
    ap.add_argument('--lr-multiplier', type=float, default=0.9)
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    device = torch.device(f'cuda:{args.gpu}')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ts = datetime.now().strftime('%d_%m_%H%M%S')
    if args.tag:
        ts += f'_{args.tag}'
    run_dir = Path(args.results_dir) / 'runs_sdf' / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f'run dir: {run_dir}')
    settings = {
        'batch_size': args.batch_size, 'clamp_value': args.clamp_value,
        'dataset': args.dataset, 'epochs': args.epochs,
        'inner_dim': args.inner_dim, 'latent_size': args.latent_size,
        'lr_latent': args.lr_latent, 'lr_model': args.lr_model,
        'lr_multiplier': args.lr_multiplier, 'num_layers': args.num_layers,
        'patience': args.patience, 'seed': args.seed,
        'sigma_regulariser': args.sigma, 'skip_connections': args.skip_connections,
    }
    with open(run_dir / 'settings.yaml', 'w') as f:
        yaml.dump(settings, f)
    samples_path = Path(args.results_dir) / f'samples_dict_{args.dataset}.npy'
    lc, co, sd, n_shapes = load_dataset(samples_path, args.clamp_value, device)
    g_cpu = torch.Generator(device='cpu')
    g_cpu.manual_seed(args.seed)
    perm = torch.randperm(len(sd), generator=g_cpu, device='cpu')
    n_train = int(0.85 * len(sd))
    train_idx = perm[:n_train].to(device)
    val_idx = perm[n_train:].to(device)
    print(f'split: {len(train_idx)} train / {len(val_idx)} val')
    model = SDFModel(num_layers=args.num_layers, skip_connections=args.skip_connections, latent_size=args.latent_size, inner_dim=args.inner_dim).float().to(device)
    latents = torch.normal(0.0, 0.01, size=(n_shapes, args.latent_size), device=device).requires_grad_(True)
    optim_m = torch.optim.Adam(model.parameters(), lr=args.lr_model)
    optim_z = torch.optim.Adam([latents], lr=args.lr_latent)
    sched_m = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_m, mode='min', factor=args.lr_multiplier, patience=args.patience, threshold=0.0001, threshold_mode='rel')
    sched_z = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_z, mode='min', factor=args.lr_multiplier, patience=args.patience, threshold=0.0001, threshold_mode='rel')
    log_csv = open(run_dir / 'loss_curve.csv', 'w', newline='')
    log_w = csv.writer(log_csv)
    log_w.writerow(['epoch', 'train_loss', 'val_loss', 'val_l1', 'val_l2', 'lr_model', 'lr_latent', 'elapsed_s'])
    best_val = float('inf')
    t0 = time.time()
    train_losses, val_losses = ([], [])
    for epoch in range(args.epochs):
        model.train()
        batches = epoch_indices(len(train_idx), args.batch_size, g_cpu)
        total = 0.0
        n_batches = 0
        for b in batches:
            sel = train_idx[b]
            z = latents[lc[sel]]
            x = torch.cat([z, co[sel]], dim=1)
            pred = model(x).squeeze(-1)
            pred = torch.clamp(pred, -args.clamp_value, args.clamp_value)
            loss, _, _ = sdf_loss(pred, sd[sel], z, args.sigma)
            optim_m.zero_grad(set_to_none=True)
            optim_z.zero_grad(set_to_none=True)
            loss.backward()
            optim_z.step()
            optim_m.step()
            total += loss.item()
            n_batches += 1
        avg_train = total / max(n_batches, 1)
        model.eval()
        with torch.no_grad():
            v_total = v_l1 = v_l2 = 0.0
            v_n = 0
            for b in epoch_indices(len(val_idx), args.batch_size, g_cpu):
                sel = val_idx[b]
                z = latents[lc[sel]]
                x = torch.cat([z, co[sel]], dim=1)
                pred = model(x).squeeze(-1)
                pred = torch.clamp(pred, -args.clamp_value, args.clamp_value)
                loss, l1, l2 = sdf_loss(pred, sd[sel], z, args.sigma)
                v_total += loss.item()
                v_l1 += l1.item()
                v_l2 += l2.item()
                v_n += 1
            avg_val = v_total / max(v_n, 1)
            avg_l1 = v_l1 / max(v_n, 1)
            avg_l2 = v_l2 / max(v_n, 1)
        sched_m.step(avg_val)
        sched_z.step(avg_val)
        lr_m = optim_m.param_groups[0]['lr']
        lr_z = optim_z.param_groups[0]['lr']
        elapsed = time.time() - t0
        log_w.writerow([epoch, f'{avg_train:.6f}', f'{avg_val:.6f}', f'{avg_l1:.6f}', f'{avg_l2:.6f}', f'{lr_m:.2e}', f'{lr_z:.2e}', f'{elapsed:.1f}'])
        log_csv.flush()
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        marker = ''
        if avg_val < best_val:
            best_val = avg_val
            marker = '  *'
            torch.save(model.state_dict(), run_dir / 'weights.pt')
            np.save(run_dir / 'results.npy', {'best_latent_codes': latents.detach().cpu().numpy()}, allow_pickle=True)
        if epoch % 10 == 0 or epoch == args.epochs - 1 or marker:
            print(f'  ep{epoch:>4d}  train={avg_train:.5f}  val={avg_val:.5f} (l1={avg_l1:.5f} l2={avg_l2:.5f})  lr_m={lr_m:.1e} lr_z={lr_z:.1e}  t={elapsed:.0f}s{marker}')
    log_csv.close()
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor='#0A0F1A')
    ax.set_facecolor('#0A0F1A')
    ax.plot(train_losses, color='#3B8FD8', label='train')
    ax.plot(val_losses, color='#E8743C', label='val')
    ax.set_xlabel('epoch', color='#A0A8B0')
    ax.set_ylabel('loss', color='#A0A8B0')
    ax.tick_params(colors='#A0A8B0')
    for s in ax.spines.values():
        s.set_color('#2A3340')
    ax.legend(facecolor='#10182A', edgecolor='#2A3340', labelcolor='#D8DEE6')
    ax.set_title(f'DeepSDF loss ({n_shapes} shapes, {args.epochs} epochs)', color='#D8DEE6')
    fig.tight_layout()
    fig.savefig(run_dir / 'loss_curve.png', dpi=110, facecolor='#0A0F1A')
    plt.close(fig)
    print(f'best val={best_val:.6f}; artefacts -> {run_dir}')
if __name__ == '__main__':
    main()
