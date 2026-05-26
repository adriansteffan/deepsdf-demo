import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from sdf_model import load_decoder


def parse_mask(spec):
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    spec = spec.strip().lower()
    if len(spec) < 3 or spec[1] not in '<>':
        raise ValueError(f"bad mask spec {spec!r} (use e.g. 'z<0')")
    axis = axis_map[spec[0]]
    sign = -1 if spec[1] == '<' else +1
    return (axis, sign)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--samples-npy', required=True)
    ap.add_argument('--target-idx', type=int, required=True)
    ap.add_argument('--mask', default='z<0')
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--dump-every', type=int, default=10)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--reg-coef', type=float, default=1.0)
    ap.add_argument('--out-dir', default='completion_traj')
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()
    device = torch.device(f'cuda:{args.gpu}')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, cfg = load_decoder(args.run_dir, device)
    all_latents = np.load(Path(args.run_dir) / 'results.npy', allow_pickle=True).item()['best_latent_codes']
    target_z = all_latents[args.target_idx].astype(np.float32)
    samples = np.load(args.samples_npy, allow_pickle=True).item()
    rec = samples[args.target_idx]
    full_sdf = rec['sdf'].reshape(-1).astype(np.float32)
    full_pts = rec['samples_latent_class'][:, 1:4].astype(np.float32)
    axis, sign = parse_mask(args.mask)
    keep = full_pts[:, axis] < 0.0 if sign < 0 else full_pts[:, axis] > 0.0
    partial_pts = full_pts[keep]
    partial_sdf = full_sdf[keep]
    print(f'{len(all_latents)} latents; target idx {args.target_idx} ||z*||={np.linalg.norm(target_z):.3f}')
    print(f'mask {args.mask}: {len(partial_pts)}/{len(full_pts)} samples kept')
    z_init = all_latents.mean(axis=0).astype(np.float32)
    z = torch.tensor(z_init, device=device).unsqueeze(0).clone().requires_grad_(True)
    p_xyz = torch.from_numpy(partial_pts).to(device)
    p_d = torch.from_numpy(partial_sdf).to(device)
    optimizer = torch.optim.Adam([z], lr=args.lr)
    clamp_value = float(cfg['clamp_value'])
    sigma = float(cfg['sigma_regulariser'])
    z_traj = [z.detach().cpu().numpy().copy().squeeze(0)]
    losses = []
    for step in range(args.steps):
        optimizer.zero_grad()
        inp = torch.cat([z.expand(p_xyz.shape[0], -1), p_xyz], dim=1)
        pred = torch.clamp(model(inp).squeeze(-1), -clamp_value, clamp_value)
        targ = torch.clamp(p_d, -clamp_value, clamp_value)
        recon = F.l1_loss(pred, targ)
        reg = sigma ** 2 * z.norm(p=2) * args.reg_coef
        loss = recon + reg
        loss.backward()
        optimizer.step()
        losses.append((loss.item(), recon.item(), reg.item()))
        if (step + 1) % args.dump_every == 0:
            z_traj.append(z.detach().cpu().numpy().copy().squeeze(0))
        if step % 50 == 0 or step == args.steps - 1:
            d_to_target = float(np.linalg.norm(z.detach().cpu().numpy().squeeze(0) - target_z))
            print(f'  step {step:4d}  recon={recon.item():.5f}  ||z-z*||={d_to_target:.3f}')
    z_traj = np.stack(z_traj).astype(np.float32)
    losses = np.asarray(losses, dtype=np.float32)
    np.save(out_dir / 'z_traj.npy', z_traj)
    np.save(out_dir / 'losses.npy', losses)
    np.save(out_dir / 'target_z.npy', target_z)
    np.save(out_dir / 'partial_points.npy', partial_pts)
    np.save(out_dir / 'partial_sdf.npy', partial_sdf)
    (out_dir / 'target_idx.txt').write_text(str(args.target_idx) + '\n')
    (out_dir / 'mask_axis.txt').write_text(args.mask + '\n')
    print(f'wrote {len(z_traj)} checkpoints; final ||z-z*||={np.linalg.norm(z_traj[-1] - target_z):.3f}')
if __name__ == '__main__':
    main()
