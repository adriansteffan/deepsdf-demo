import argparse
import os
from pathlib import Path
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import numpy as np
import torch
import trimesh
import imageio.v2 as imageio
import pyrender
from skimage import measure
from PIL import Image, ImageDraw, ImageFont

from sdf_model import load_decoder


def look_at(eye, target, up):
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    M = np.eye(4, dtype=np.float32)
    M[:3, 0] = s
    M[:3, 1] = u
    M[:3, 2] = -f
    M[:3, 3] = eye
    return M

def decode_to_mesh(model, z, resolution, device, max_pts=1 << 18):
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
    ap.add_argument('--out-dir', default='all_anchors')
    ap.add_argument('--resolution', type=int, default=64)
    ap.add_argument('--cell-size', type=int, default=200)
    ap.add_argument('--cols', type=int, default=15)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()
    device = torch.device(f'cuda:{args.gpu}')
    run_dir = Path(args.run_dir)
    model, _ = load_decoder(run_dir, device)
    latents = np.load(run_dir / 'results.npy', allow_pickle=True).item()['best_latent_codes']
    n = len(latents)
    print(f'decoding {n} latents @ {args.resolution}^3')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    cell_dir = out_dir / 'cells'
    cell_dir.mkdir(exist_ok=True)
    W = H = args.cell_size
    material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.86, 0.8, 0.72, 1.0], metallicFactor=0.0, roughnessFactor=0.55)
    scene = pyrender.Scene(bg_color=[0.04, 0.06, 0.1, 1.0], ambient_light=[0.27, 0.27, 0.29])
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=4.0), pose=look_at([2.0, 2.0, 2.5], [0, 0, 0], [0, 0, 1]))
    scene.add(pyrender.DirectionalLight(color=np.ones(3) * 0.85, intensity=2.0), pose=look_at([-2.5, -1.0, 1.5], [0, 0, 0], [0, 0, 1]))
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=W / H)
    scene.add(cam, pose=look_at([1.4, -1.4, 0.6], [0, 0, 0], [0, 0, 1]))
    renderer = pyrender.OffscreenRenderer(W, H)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 14)
    except Exception:
        font = ImageFont.load_default()
    cells = []
    n_empty = 0
    for i, z in enumerate(latents):
        mesh = decode_to_mesh(model, z.astype(np.float32), args.resolution, device)
        if mesh is None:
            n_empty += 1
            img = np.full((H, W, 3), (10, 16, 26), dtype=np.uint8)
        else:
            pmesh = pyrender.Mesh.from_trimesh(mesh, smooth=True, material=material)
            node = scene.add(pmesh)
            color, _ = renderer.render(scene)
            scene.remove_node(node)
            img = color
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        draw.text((6, H - 22), f'#{i}', fill='#FFFFFF', font=font, stroke_width=1, stroke_fill='#000000')
        img = np.asarray(pil)
        imageio.imwrite(cell_dir / f'anchor_{i:04d}.png', img)
        cells.append(img)
        if i % 20 == 0 or i == n - 1:
            print(f'  {i + 1}/{n}')
    cols = args.cols
    rows = (n + cols - 1) // cols
    sheet = np.full((rows * H, cols * W, 3), (10, 16, 26), dtype=np.uint8)
    for i, img in enumerate(cells):
        r, c = (i // cols, i % cols)
        sheet[r * H:(r + 1) * H, c * W:(c + 1) * W] = img
    imageio.imwrite(out_dir / 'contact_sheet.png', sheet)
    print(f"wrote {out_dir / 'contact_sheet.png'} ({n_empty} empty)")
    renderer.delete()
if __name__ == '__main__':
    main()
