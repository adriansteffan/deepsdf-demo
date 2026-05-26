import argparse
import time
from pathlib import Path
import numpy as np
import torch
import trimesh
from skimage import measure

from sdf_model import load_decoder


def smoothstep(t):
    return t * t * (3.0 - 2.0 * t)

def smoothstep_overall_path(z_traj, n_frames):
    K = len(z_traj)
    out = []
    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        s = smoothstep(t)
        u = s * (K - 1)
        seg = int(np.floor(u))
        if seg >= K - 1:
            out.append(z_traj[-1].copy())
        else:
            local = u - seg
            out.append((1 - local) * z_traj[seg] + local * z_traj[seg + 1])
    return np.stack(out).astype(np.float32)

def decode_mesh(model, z, resolution, device, max_pts=1 << 18):
    grid = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    xx, yy, zz = np.meshgrid(grid, grid, grid, indexing='ij')
    pts = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    pts_t = torch.from_numpy(pts).to(device)
    z_t = torch.from_numpy(z).to(device)
    sdf = torch.empty(pts_t.shape[0], device=device)
    with torch.no_grad():
        for i in range(0, pts_t.shape[0], max_pts):
            chunk = pts_t[i:i + max_pts]
            inp = torch.cat([z_t.unsqueeze(0).expand(chunk.shape[0], -1), chunk], dim=1)
            sdf[i:i + max_pts] = model(inp).squeeze(-1)
    grid_sdf = sdf.cpu().numpy().reshape(resolution, resolution, resolution)
    try:
        verts, faces, _, _ = measure.marching_cubes(grid_sdf, level=0.0, spacing=(2.0 / resolution,) * 3)
    except (ValueError, RuntimeError):
        return None
    verts -= 1.0
    return trimesh.Trimesh(vertices=verts, faces=faces)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--traj-dir', default='completion_traj')
    ap.add_argument('--out-dir', default='completion_meshes')
    ap.add_argument('--n-frames', type=int, default=240)
    ap.add_argument('--resolution', type=int, default=256)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()
    traj_dir = Path(args.traj_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}')
    model, _ = load_decoder(args.run_dir, device)
    z_traj = np.load(traj_dir / 'z_traj.npy')
    target_z = np.load(traj_dir / 'target_z.npy')
    gt_mesh = decode_mesh(model, target_z.astype(np.float32), args.resolution, device)
    if gt_mesh is None:
        raise SystemExit('GT decode produced empty mesh')
    gt_mesh.export(out_dir / 'gt.obj')
    z_path = smoothstep_overall_path(z_traj, args.n_frames)
    np.save(out_dir / 'z_path.npy', z_path)
    np.save(out_dir / 'z_traj_used.npy', z_traj.astype(np.float32))
    t0 = time.time()
    n_empty = 0
    for i, z in enumerate(z_path):
        mesh = decode_mesh(model, z.astype(np.float32), args.resolution, device)
        if mesh is None:
            n_empty += 1
            continue
        mesh.export(out_dir / f'frame_{i:04d}.obj')
        if i % 30 == 0 or i == args.n_frames - 1:
            dt = time.time() - t0
            eta = dt / (i + 1) * (args.n_frames - i - 1)
            print(f'  frame {i + 1:>4d}/{args.n_frames}  V={len(mesh.vertices):>5d}  t={dt:.0f}s eta={eta:.0f}s')
    print(f'{args.n_frames} frames in {time.time() - t0:.0f}s ({n_empty} empty)')
if __name__ == '__main__':
    main()
