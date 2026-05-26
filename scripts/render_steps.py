import argparse
import os
from pathlib import Path
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import numpy as np
import torch
import trimesh
import yaml
import imageio.v2 as imageio
import pyrender
from PIL import Image
from skimage import measure

from sdf_model import SDFModel


def load_release(model_dir, device):
    model_dir = Path(model_dir)
    cfg = yaml.safe_load(open(model_dir / 'settings.yaml'))
    model = SDFModel(cfg['num_layers'], cfg['skip_connections'],
                     cfg['latent_size'], cfg['inner_dim']).to(device)
    model.load_state_dict(torch.load(model_dir / 'weights.pt',
                                     map_location=device, weights_only=False))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    latents = np.load(model_dir / 'results.npy', allow_pickle=True).item()['best_latent_codes']
    return model, latents


def decode_mesh(model, z, resolution, device, max_pts=1 << 18):
    grid = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    xx, yy, zz = np.meshgrid(grid, grid, grid, indexing='ij')
    pts = torch.from_numpy(np.stack([xx, yy, zz], -1).reshape(-1, 3)).to(device)
    zt = torch.from_numpy(z.astype(np.float32)).to(device)
    sdf = torch.empty(pts.shape[0], device=device)
    for i in range(0, pts.shape[0], max_pts):
        chunk = pts[i:i + max_pts]
        sdf[i:i + max_pts] = model(torch.cat([zt.unsqueeze(0).expand(chunk.shape[0], -1), chunk], 1)).squeeze(-1)
    g = sdf.cpu().numpy().reshape(resolution, resolution, resolution)
    v, f, _, _ = measure.marching_cubes(g, level=0.0, spacing=(2.0 / resolution,) * 3)
    return trimesh.Trimesh(vertices=v - 1.0, faces=f, process=False)


def look_at(eye, target, up):
    eye, target, up = (np.asarray(a, dtype=np.float32) for a in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    M = np.eye(4, dtype=np.float32)
    M[:3, 0] = s; M[:3, 1] = u; M[:3, 2] = -f; M[:3, 3] = eye
    return M


def build_scene(W, H):
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.45, 0.45, 0.45, 1.0], metallicFactor=0.0, roughnessFactor=0.75)
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.20, 0.20, 0.20])
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.5),
              pose=look_at([2.0, 2.0, 2.5], [0, 0, 0], [0, 0, 1]))
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.2),
              pose=look_at([-2.5, -1.0, 1.5], [0, 0, 0], [0, 0, 1]))
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=W / H)
    cam_node = scene.add(cam, pose=np.eye(4))
    renderer = pyrender.OffscreenRenderer(W, H)
    return scene, cam_node, renderer, material


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', default='model/run_shipped')
    ap.add_argument('--out-dir', default='figures/morph_steps')
    ap.add_argument('--from-idx', type=int, default=111)
    ap.add_argument('--to-idx', type=int, default=120)
    ap.add_argument('--n-steps', type=int, default=7,
                    help='Total frames including endpoints (5 intermediates -> 7)')
    ap.add_argument('--resolution', type=int, default=256)
    ap.add_argument('--width', type=int, default=1024)
    ap.add_argument('--height', type=int, default=1024)
    ap.add_argument('--cam-distance', type=float, default=2.35)
    ap.add_argument('--cam-height', type=float, default=0.95)
    ap.add_argument('--target-extent', type=float, default=1.45,
                    help='Diagonal of each mesh\'s projected screen-space bbox after scaling')
    ap.add_argument('--azimuth-deg', type=float, default=135.0)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--strip', action='store_true', default=True,
                    help='Also write a horizontal strip of all frames')
    ap.add_argument('--strip-height', type=int, default=490)
    ap.add_argument('--strip-aspect', type=float, default=2475 / 490)
    ap.add_argument('--strip-gap', type=int, default=10)
    args = ap.parse_args()
    device = torch.device(f'cuda:{args.gpu}')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, latents = load_release(args.model_dir, device)
    z_a = latents[args.from_idx].astype(np.float32)
    z_b = latents[args.to_idx].astype(np.float32)
    ts = np.linspace(0.0, 1.0, args.n_steps)
    print(f'rendering {args.n_steps} frames idx {args.from_idx} -> {args.to_idx} '
          f'at t={list(np.round(ts, 3))}')

    scene, cam_node, renderer, material = build_scene(args.width, args.height)
    theta = np.deg2rad(args.azimuth_deg)
    eye = np.array([args.cam_distance * np.cos(theta),
                    -args.cam_distance * np.sin(theta),
                    args.cam_height], dtype=np.float32)
    scene.set_pose(cam_node, look_at(eye, [0, 0, 0], [0, 0, 1]))

    forward = -eye / np.linalg.norm(eye)
    cam_right = np.cross(forward, [0, 0, 1]); cam_right /= np.linalg.norm(cam_right)
    cam_up = np.cross(cam_right, forward)

    rendered = []
    for i, t in enumerate(ts):
        z = (1.0 - t) * z_a + t * z_b
        mesh = decode_mesh(model, z, args.resolution, device)
        v = np.asarray(mesh.vertices)
        center = (v.min(0) + v.max(0)) * 0.5
        vc = v - center
        proj_x = vc @ cam_right
        proj_y = vc @ cam_up
        sx = float(proj_x.max() - proj_x.min())
        sy = float(proj_y.max() - proj_y.min())
        diag = float(np.hypot(sx, sy))
        mesh.vertices = vc * (args.target_extent / diag)
        pm = pyrender.Mesh.from_trimesh(mesh, smooth=True, material=material)
        node = scene.add(pm)
        color, _ = renderer.render(scene)
        scene.remove_node(node)
        out_path = out_dir / f'step_{i:02d}_t{t:.3f}.png'
        imageio.imwrite(out_path, color)
        rendered.append(color)
        print(f'  [{i + 1}/{args.n_steps}] t={t:.3f}  V={len(mesh.vertices):>5d}  -> {out_path}')
    renderer.delete()

    if args.strip:
        H = args.strip_height
        gap = args.strip_gap
        n = args.n_steps
        target_W = int(round(H * args.strip_aspect))
        panel_w = (target_W - gap * (n - 1)) // n
        W = panel_w * n + gap * (n - 1)
        crop_ratio = panel_w / H
        strip = np.full((H, W, 3), 255, dtype=np.uint8)
        for i, img in enumerate(rendered):
            ih, iw = img.shape[:2]
            crop_w = int(round(ih * crop_ratio))
            x0 = (iw - crop_w) // 2
            cropped = img[:, x0:x0 + crop_w]
            resized = np.asarray(Image.fromarray(cropped).resize((panel_w, H), Image.LANCZOS))
            x = i * (panel_w + gap)
            strip[:, x:x + panel_w] = resized
        strip_path = out_dir / 'strip.png'
        imageio.imwrite(strip_path, strip)
        print(f'strip: {W}x{H} (aspect {W / H:.3f}) -> {strip_path}')


if __name__ == '__main__':
    main()
