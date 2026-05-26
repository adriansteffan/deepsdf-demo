import argparse
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import point_cloud_utils as pcu
import fast_simplification
from skimage import measure

from sdf_model import load_decoder


def decimate(mesh, target_faces):
    verts, faces = fast_simplification.simplify(np.asarray(mesh.vertices), np.asarray(mesh.faces), target_count=target_faces)
    out = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return out

def sample_sdf_from_mesh(verts, faces, n_vol=3000, n_bbox=10000, n_surf=10000, seed=42):
    np.random.seed(seed)
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    p_vol = np.random.rand(n_vol, 3).astype(np.float32) * 2 - 1
    v_min = verts.min(0)
    v_max = verts.max(0)
    p_bbox = np.random.uniform(low=v_min, high=v_max, size=(n_bbox, 3)).astype(np.float32)
    fid_surf, bc_surf = pcu.sample_mesh_random(verts, faces, n_surf)
    p_surf = pcu.interpolate_barycentric_coords(faces, fid_surf, bc_surf, verts)
    p_total = np.vstack((p_vol, p_bbox, p_surf)).astype(np.float32)
    sdf, _, _ = pcu.signed_distance_to_mesh(p_total, verts, faces)
    return (p_total, sdf.astype(np.float32))

def optimize_z(model, target_pts, target_sdf, z_init, lr, steps, clamp_value, sigma, reg_coef, device, dump_every=10):
    z = torch.tensor(z_init).float().to(device).unsqueeze(0).clone().requires_grad_(True)
    p_xyz = torch.from_numpy(target_pts).to(device)
    p_d = torch.from_numpy(target_sdf).to(device)
    opt = torch.optim.Adam([z], lr=lr)
    z_traj = [z.detach().squeeze().cpu().numpy().copy()]
    for step in range(steps):
        opt.zero_grad()
        inp = torch.cat([z.expand(p_xyz.shape[0], -1), p_xyz], dim=1)
        pred = model(inp).squeeze(-1)
        pred_c = torch.clamp(pred, -clamp_value, clamp_value)
        targ_c = torch.clamp(p_d, -clamp_value, clamp_value)
        recon = F.l1_loss(pred_c, targ_c)
        reg = sigma ** 2 * z.norm(p=2) * reg_coef
        loss = recon + reg
        loss.backward()
        opt.step()
        if (step + 1) % dump_every == 0:
            z_traj.append(z.detach().squeeze().cpu().numpy().copy())
        if step % 50 == 0 or step == steps - 1:
            print(f'  step {step:4d}  loss={loss.item():.5f}  recon={recon.item():.5f}  ||z||={z.norm().item():.3f}')
    return (z.detach().squeeze().cpu().numpy(), np.stack(z_traj).astype(np.float32))

def decode_mesh(model, z, resolution, device, max_pts=1 << 18):
    grid = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    xx, yy, zz = np.meshgrid(grid, grid, grid, indexing='ij')
    pts = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    pts_t = torch.from_numpy(pts).to(device)
    z_t = torch.from_numpy(z.astype(np.float32)).to(device)
    sdf = torch.empty(pts_t.shape[0], device=device)
    with torch.no_grad():
        for i in range(0, pts_t.shape[0], max_pts):
            chunk = pts_t[i:i + max_pts]
            inp = torch.cat([z_t.unsqueeze(0).expand(chunk.shape[0], -1), chunk], dim=1)
            sdf[i:i + max_pts] = model(inp).squeeze(-1)
    grid_sdf = sdf.cpu().numpy().reshape(resolution, resolution, resolution)
    verts, faces, _, _ = measure.marching_cubes(grid_sdf, level=0.0, spacing=(2 / resolution,) * 3)
    verts -= 1.0
    return trimesh.Trimesh(vertices=verts, faces=faces)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--obj-path', required=True)
    ap.add_argument('--target-faces', type=int, default=500)
    ap.add_argument('--out-dir', default='upscale_demo')
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--lr', type=float, default=0.005)
    ap.add_argument('--reg-coef', type=float, default=50.0)
    ap.add_argument('--resolution', type=int, default=256)
    ap.add_argument('--init-z-idx', type=int, default=None)
    ap.add_argument('--dump-every', type=int, default=10)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0')
    orig = trimesh.load(args.obj_path, process=False)
    if not orig.is_watertight:
        v, f = pcu.make_mesh_watertight(np.asarray(orig.vertices, dtype=np.float64),
                                         np.asarray(orig.faces, dtype=np.int32), 50000)
        orig = trimesh.Trimesh(vertices=v, faces=f, process=False)
    low = decimate(orig, args.target_faces)
    low.export(out_dir / 'input_lowres.obj')
    verts = np.asarray(low.vertices, dtype=np.float64)
    faces = np.asarray(low.faces, dtype=np.int32)
    if not low.is_watertight:
        verts, faces = pcu.make_mesh_watertight(verts, faces, 50000)
    pts, sdf = sample_sdf_from_mesh(verts, faces)
    print(f'input: {len(low.vertices)}V / {len(low.faces)}F decimated; {len(pts)} SDF samples')
    model, cfg = load_decoder(args.run_dir, device)
    all_latents = np.load(Path(args.run_dir) / 'results.npy', allow_pickle=True).item()['best_latent_codes']
    if args.init_z_idx is not None:
        z_init = all_latents[args.init_z_idx].astype(np.float32)
    else:
        z_init = all_latents.mean(axis=0).astype(np.float32)
    t0 = time.time()
    z_final, z_traj = optimize_z(model, pts, sdf, z_init, args.lr, args.steps,
                                  float(cfg['clamp_value']), float(cfg['sigma_regulariser']),
                                  args.reg_coef, device, dump_every=args.dump_every)
    print(f'optim: {time.time() - t0:.1f}s, {len(z_traj)} checkpoints, ||z||={np.linalg.norm(z_final):.3f}')
    np.save(out_dir / 'z_final.npy', z_final)
    np.save(out_dir / 'z_traj.npy', z_traj)
    np.save(out_dir / 'partial_points.npy', pts)
    np.save(out_dir / 'partial_sdf.npy', sdf)
    t0 = time.time()
    upscaled = decode_mesh(model, z_final, args.resolution, device)
    upscaled.export(out_dir / 'upscaled.obj')
    print(f'decoded @ {args.resolution}^3 in {time.time() - t0:.1f}s: {len(upscaled.vertices)}V / {len(upscaled.faces)}F')
if __name__ == '__main__':
    main()
