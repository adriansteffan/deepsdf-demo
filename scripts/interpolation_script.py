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

def make_path(codes, num_frames, easing='smoothstep'):
    codes = list(codes)
    n_seg = len(codes) - 1
    if num_frames < n_seg + 1:
        raise ValueError('num_frames must allow at least 1 frame per segment')
    path = []
    for i in range(num_frames):
        u = i / (num_frames - 1) * n_seg
        seg = int(np.floor(u))
        if seg >= n_seg:
            seg = n_seg - 1
            t = 1.0
        else:
            t = u - seg
        s = smoothstep(t) if easing == 'smoothstep' else t
        a, b = (codes[seg], codes[seg + 1])
        path.append((1 - s) * a + s * b)
    return np.stack(path).astype(np.float32)

def create_mesh(model, z, resolution, device, max_pts=1 << 18):
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
    ap.add_argument('--resolution', type=int, default=128)
    ap.add_argument('--num-frames', type=int, default=600)
    ap.add_argument('--easing', choices=['linear', 'smoothstep'], default='smoothstep')
    ap.add_argument('--out-dir', type=str, default='frames_obj')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--anchors-only', action='store_true')
    ap.add_argument('--latent-indices', default=None)
    args = ap.parse_args()
    device = torch.device(f'cuda:{args.gpu}')
    run_dir = Path(args.run_dir)
    model, _ = load_decoder(run_dir, device)
    all_latents = np.load(run_dir / 'results.npy', allow_pickle=True).item()['best_latent_codes']
    if args.latent_indices:
        idxs = [int(s) for s in args.latent_indices.split(',')]
        latents = all_latents[idxs]
    else:
        latents = all_latents
    print(f'{len(all_latents)} latents loaded; using {len(latents)} as anchors')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.anchors_only:
        anchor_dir = out_dir / 'anchors'
        anchor_dir.mkdir(exist_ok=True)
        for i, z in enumerate(latents):
            mesh = create_mesh(model, z.astype(np.float32), args.resolution, device)
            if mesh is None:
                print(f'anchor {i}: empty')
                continue
            mesh.export(anchor_dir / f'anchor_{i}.obj')
        return
    all_zs = make_path(list(latents), num_frames=args.num_frames, easing=args.easing)
    np.save(out_dir / 'path.npy', all_zs)
    np.save(out_dir / 'anchors.npy', np.asarray(latents, dtype=np.float32))
    t_total = time.time()
    n_empty = 0
    for i, z in enumerate(all_zs):
        mesh = create_mesh(model, z, args.resolution, device)
        if mesh is None:
            n_empty += 1
            continue
        mesh.export(out_dir / f'frame_{i:04d}.obj')
        if i % 50 == 0 or i == len(all_zs) - 1:
            print(f'  frame {i:04d}/{len(all_zs)}  V={len(mesh.vertices)}')
    print(f'{len(all_zs)} frames in {time.time() - t_total:.1f}s ({n_empty} empty)')
if __name__ == '__main__':
    main()
