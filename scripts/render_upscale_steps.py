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
    return model


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
    try:
        v, f, _, _ = measure.marching_cubes(g, level=0.0, spacing=(2.0 / resolution,) * 3)
    except (ValueError, RuntimeError):
        return None
    m = trimesh.Trimesh(vertices=v - 1.0, faces=f, process=False)
    parts = m.split(only_watertight=False)
    if len(parts) > 1:
        m = max(parts, key=lambda c: len(c.faces))
    return m


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
    ap.add_argument('--traj', default='model/z_traj.npy')
    ap.add_argument('--input-obj', default='model/input_lowres.obj',
                    help='Low-res input mesh shown as the leftmost panel; pass "" to skip')
    ap.add_argument('--out-dir', default='figures/upscale_steps')
    ap.add_argument('--n-steps', type=int, default=7,
                    help='Total panels in the strip; if --input-obj is set, first panel is the input mesh and the rest are decoded latents')
    ap.add_argument('--mode', choices=['traj', 'interp'], default='interp',
                    help='interp = linear interp z[0]->z[-1]; traj = sample equispaced from z_traj.npy (Adam converges in ~8 steps so most frames look identical)')
    ap.add_argument('--resolution', type=int, default=256)
    ap.add_argument('--width', type=int, default=1024)
    ap.add_argument('--height', type=int, default=1024)
    ap.add_argument('--cam-distance', type=float, default=2.35)
    ap.add_argument('--cam-height', type=float, default=0.95)
    ap.add_argument('--azimuth-deg', type=float, default=135.0)
    ap.add_argument('--target-extent', type=float, default=1.45)
    ap.add_argument('--strip-height', type=int, default=490)
    ap.add_argument('--strip-aspect', type=float, default=2475 / 490)
    ap.add_argument('--strip-gap', type=int, default=10)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()
    device = torch.device(f'cuda:{args.gpu}')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_release(args.model_dir, device)
    traj = np.load(args.traj)
    print(f'loaded z_traj: {traj.shape}, ||z[0]||={np.linalg.norm(traj[0]):.3f}, ||z[-1]||={np.linalg.norm(traj[-1]):.3f}')
    input_mesh = trimesh.load(args.input_obj, process=False) if args.input_obj else None
    n_latents = args.n_steps - (1 if input_mesh is not None else 0)
    if args.mode == 'traj':
        picks = np.linspace(0, len(traj) - 1, n_latents).round().astype(int)
        zs = traj[picks]
        print(f'picked checkpoints {picks.tolist()}')
    else:
        ts = np.linspace(0.0, 1.0, n_latents)
        zs = np.stack([(1.0 - t) * traj[0] + t * traj[-1] for t in ts])
        print(f'interp t={list(np.round(ts, 3))}')

    scene, cam_node, renderer, material = build_scene(args.width, args.height)
    theta = np.deg2rad(args.azimuth_deg)
    eye = np.array([args.cam_distance * np.cos(theta),
                    -args.cam_distance * np.sin(theta),
                    args.cam_height], dtype=np.float32)
    scene.set_pose(cam_node, look_at(eye, [0, 0, 0], [0, 0, 1]))

    forward = -eye / np.linalg.norm(eye)
    cam_right = np.cross(forward, [0, 0, 1]); cam_right /= np.linalg.norm(cam_right)
    cam_up = np.cross(cam_right, forward)

    def render_mesh(mesh, smooth=True):
        v = np.asarray(mesh.vertices)
        center = (v.min(0) + v.max(0)) * 0.5
        vc = v - center
        sx = float((vc @ cam_right).max() - (vc @ cam_right).min())
        sy = float((vc @ cam_up).max() - (vc @ cam_up).min())
        diag = float(np.hypot(sx, sy))
        m = trimesh.Trimesh(vertices=vc * (args.target_extent / diag),
                            faces=np.asarray(mesh.faces), process=False)
        pm = pyrender.Mesh.from_trimesh(m, smooth=smooth, material=material)
        node = scene.add(pm)
        color, _ = renderer.render(scene)
        scene.remove_node(node)
        return color

    rendered = []
    panel_idx = 0
    if input_mesh is not None:
        color = render_mesh(input_mesh, smooth=False)
        out_path = out_dir / f'step_{panel_idx:02d}_input.png'
        imageio.imwrite(out_path, color)
        rendered.append(color)
        print(f'  [{panel_idx + 1}/{args.n_steps}] INPUT  V={len(input_mesh.vertices)} F={len(input_mesh.faces)}  -> {out_path}')
        panel_idx += 1

    for z in zs:
        mesh = decode_mesh(model, z.astype(np.float32), args.resolution, device)
        if mesh is None or len(mesh.vertices) < 10:
            print(f'  [{panel_idx + 1}/{args.n_steps}] EMPTY decode (||z||={np.linalg.norm(z):.3f})')
            rendered.append(np.full((args.height, args.width, 3), 255, dtype=np.uint8))
            panel_idx += 1
            continue
        color = render_mesh(mesh)
        out_path = out_dir / f'step_{panel_idx:02d}.png'
        imageio.imwrite(out_path, color)
        rendered.append(color)
        print(f'  [{panel_idx + 1}/{args.n_steps}] ||z||={np.linalg.norm(z):.3f}  V={len(mesh.vertices):>5d}  -> {out_path}')
        panel_idx += 1
    renderer.delete()

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
