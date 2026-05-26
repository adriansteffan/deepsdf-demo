import argparse
import glob
import os
import time
from pathlib import Path
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import numpy as np
import trimesh
import imageio.v2 as imageio
import pyrender
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ANCHOR_COLORS = ['#E8743C', '#3B8FD8', '#2EAA6E', '#C26AC9', '#E8C547', '#5BCBD5']

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

def fit_pca_2d(points):
    mean = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - mean, full_matrices=False)
    return (mean, Vt[:2])

def project_pca(points, mean, basis):
    return (points - mean) @ basis.T

class LatentPanel:

    def __init__(self, anchors, path, labels, population=None, width=960, height=720, trail_len=40):
        self.anchors = anchors
        self.path = path
        self.population = population
        self.labels = labels
        self.width = width
        self.height = height
        self.trail_len = trail_len
        basis_pts = population if population is not None else np.vstack([anchors, path])
        self.mean, self.basis = fit_pca_2d(basis_pts)
        self.anchors_2d = project_pca(anchors, self.mean, self.basis)
        self.path_2d = project_pca(path, self.mean, self.basis)
        self.population_2d = project_pca(population, self.mean, self.basis) if population is not None else None
        viz_stack = [self.anchors_2d, self.path_2d]
        if self.population_2d is not None:
            viz_stack.append(self.population_2d)
        all_pts = np.vstack(viz_stack)
        xy_min = all_pts.min(axis=0)
        xy_max = all_pts.max(axis=0)
        pad = 0.1 * (xy_max - xy_min).max()
        self.xlim = (xy_min[0] - pad, xy_max[0] + pad)
        self.ylim = (xy_min[1] - pad - 0.03, xy_max[1] + pad)
        self.fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor='#0A0F1A')
        self.ax = self.fig.add_subplot(111, facecolor='#0A0F1A')
        self._setup_static()

    def _setup_static(self):
        ax = self.ax
        ax.set_xlim(*self.xlim)
        ax.set_ylim(*self.ylim)
        ax.set_aspect('equal')
        ax.tick_params(colors='#A0A8B0', labelsize=9)
        for spine in ax.spines.values():
            spine.set_color('#2A3340')
        ax.grid(True, color='#1B2330', linestyle='-', linewidth=0.7, zorder=0)
        if self.population_2d is not None:
            ax.scatter(self.population_2d[:, 0], self.population_2d[:, 1], s=18, c='#6A7C95', alpha=0.45, linewidths=0, zorder=2)
        for i in range(len(self.anchors_2d) - 1):
            a, b = (self.anchors_2d[i], self.anchors_2d[i + 1])
            ax.plot([a[0], b[0]], [a[1], b[1]], color='#5A6878', linewidth=1.4, linestyle='--', zorder=3)
        for xy, label, color in zip(self.anchors_2d, self.labels, ANCHOR_COLORS):
            ax.scatter([xy[0]], [xy[1]], s=180, c=color, edgecolors='white', linewidths=1.5, zorder=4)
            oy = (self.ylim[1] - self.ylim[0]) * 0.05
            ax.annotate(label, xy=xy, xytext=(xy[0], xy[1] + oy), ha='center', va='bottom', color=color, fontsize=14, fontweight='bold', zorder=5)
        ax.set_xlabel('PC1', color='#A0A8B0', fontsize=11)
        ax.set_ylabel('PC2', color='#A0A8B0', fontsize=11)
        d = self.anchors.shape[1]
        if self.population_2d is not None:
            title = f'Latent-space interpolation (2D PCA of {len(self.population)} trained latents, {d}-D each)'
        else:
            title = f'Latent-space interpolation (PCA of {d}-D path)'
        ax.set_title(title, color='#D8DEE6', fontsize=13, pad=12)
        self.trail_line, = ax.plot([], [], color='#FFFFFF', linewidth=2.0, alpha=0.6, zorder=6)
        self.dot = ax.scatter([], [], s=160, c='white', edgecolors='#1B2330', linewidths=2.0, zorder=7)

    def render(self, frame_idx):
        if self.trail_len > 0:
            lo = max(0, frame_idx - self.trail_len)
            seg = self.path_2d[lo:frame_idx + 1]
            self.trail_line.set_data(seg[:, 0], seg[:, 1])
        else:
            self.trail_line.set_data([], [])
        self.dot.set_offsets(self.path_2d[frame_idx:frame_idx + 1])
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    def close(self):
        plt.close(self.fig)

def build_3d_scene(panel_width, panel_height):
    material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.86, 0.8, 0.72, 1.0], metallicFactor=0.0, roughnessFactor=0.55)
    scene = pyrender.Scene(bg_color=[0.04, 0.06, 0.1, 1.0], ambient_light=[0.27, 0.27, 0.29])
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=4.0), pose=look_at([2.0, 2.0, 2.5], [0, 0, 0], [0, 0, 1]))
    scene.add(pyrender.DirectionalLight(color=np.ones(3) * 0.85, intensity=2.0), pose=look_at([-2.5, -1.0, 1.5], [0, 0, 0], [0, 0, 1]))
    scene.add(pyrender.DirectionalLight(color=np.array([0.7, 0.85, 1.0]), intensity=2.5), pose=look_at([0.0, -2.5, 1.5], [0, 0, 0], [0, 0, 1]))
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=panel_width / panel_height)
    cam_node = scene.add(cam, pose=np.eye(4))
    renderer = pyrender.OffscreenRenderer(panel_width, panel_height)
    return (scene, cam_node, renderer, material)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames-dir', required=True)
    ap.add_argument('--out-png-dir', default='frames_png')
    ap.add_argument('--out-mp4', default='morph.mp4')
    ap.add_argument('--panel-width', type=int, default=960)
    ap.add_argument('--panel-height', type=int, default=1080)
    ap.add_argument('--fps', type=int, default=60)
    ap.add_argument('--cam-distance', type=float, default=1.7)
    ap.add_argument('--cam-height', type=float, default=0.35)
    ap.add_argument('--rotation-deg', type=float, default=180.0)
    ap.add_argument('--start-angle-deg', type=float, default=0.0)
    ap.add_argument('--labels', default=None)
    ap.add_argument('--all-latents', default=None)
    ap.add_argument('--panel-order', choices=['shape-latent', 'latent-shape'], default='latent-shape')
    ap.add_argument('--trail-len', type=int, default=0)
    ap.add_argument('--loop', action='store_true')
    ap.add_argument('--hold-frames', type=int, default=0)
    args = ap.parse_args()
    obj_files = sorted(glob.glob(str(Path(args.frames_dir) / 'frame_*.obj')))
    if not obj_files:
        raise SystemExit(f'no frame_*.obj files in {args.frames_dir}')
    frame_indices = [int(Path(p).stem.split('_')[1]) for p in obj_files]
    anchors = np.load(Path(args.frames_dir) / 'anchors.npy')
    path = np.load(Path(args.frames_dir) / 'path.npy')
    if max(frame_indices) >= len(path):
        raise SystemExit(f'frame index {max(frame_indices)} out of range for path.npy with {len(path)} entries')
    out_png_dir = Path(args.out_png_dir)
    out_png_dir.mkdir(parents=True, exist_ok=True)
    scene, cam_node, renderer_3d, material = build_3d_scene(args.panel_width, args.panel_height)
    if args.labels:
        labels = [s.strip() for s in args.labels.split(',')]
        if len(labels) != len(anchors):
            raise SystemExit(f'--labels has {len(labels)} entries but anchors has {len(anchors)}')
    else:
        labels = [f'anchor {i}' for i in range(len(anchors))]
    population = None
    if args.all_latents:
        d = np.load(args.all_latents, allow_pickle=True)
        population = d.item()['best_latent_codes'] if d.dtype == object else d
        print(f'loaded {len(population)} background latents from {args.all_latents}')
    latent_panel = LatentPanel(anchors=anchors, path=path, labels=labels, population=population, width=args.panel_width, height=args.panel_height, trail_len=args.trail_len)
    if args.hold_frames > 0:
        anchor_idx_set = set()
        for a in anchors:
            anchor_idx_set.add(int(np.argmin(np.linalg.norm(path - a, axis=1))))
        held_obj, held_fi = ([], [])
        for f, fi in zip(obj_files, frame_indices):
            held_obj.append(f)
            held_fi.append(fi)
            if fi in anchor_idx_set:
                held_obj.extend([f] * args.hold_frames)
                held_fi.extend([fi] * args.hold_frames)
        print(f'holds: inserted {len(held_obj) - len(obj_files)} extra frames at {len(anchor_idx_set)} anchors ({args.hold_frames} per anchor)')
        obj_files, frame_indices = (held_obj, held_fi)
    if args.loop:
        schedule_obj = obj_files + obj_files[-2::-1]
        schedule_fi = frame_indices + frame_indices[-2::-1]
        if args.rotation_deg == 180.0:
            args.rotation_deg = 360.0
    else:
        schedule_obj = obj_files
        schedule_fi = frame_indices
    n = len(schedule_obj)
    rot_total = np.radians(args.rotation_deg)
    start_offset = np.radians(args.start_angle_deg)
    print(f'rendering {n} frames @ {2 * args.panel_width}x{args.panel_height}, fps={args.fps}, loop={args.loop}, rotation={args.rotation_deg}°')
    writer = imageio.get_writer(args.out_mp4, fps=args.fps, codec='libx264', quality=7, macro_block_size=1)
    mesh_node = None
    t_start = time.time()
    for i, (obj_path, fi) in enumerate(zip(schedule_obj, schedule_fi)):
        theta = start_offset - rot_total / 2 + rot_total * (i / max(n - 1, 1))
        eye = np.array([args.cam_distance * np.sin(theta), -args.cam_distance * np.cos(theta), args.cam_height], dtype=np.float32)
        scene.set_pose(cam_node, look_at(eye, [0, 0, 0], [0, 0, 1]))
        mesh = trimesh.load(obj_path, process=False)
        pmesh = pyrender.Mesh.from_trimesh(mesh, smooth=True, material=material)
        if mesh_node is not None:
            scene.remove_node(mesh_node)
        mesh_node = scene.add(pmesh)
        color_3d, _ = renderer_3d.render(scene)
        color_lat = latent_panel.render(fi)
        if args.panel_order == 'shape-latent':
            combined = np.hstack([color_3d, color_lat])
        else:
            combined = np.hstack([color_lat, color_3d])
        imageio.imwrite(out_png_dir / f'frame_{i:04d}.png', combined)
        writer.append_data(combined)
        if i % 50 == 0 or i == n - 1:
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (n - i - 1)
            print(f'  frame {i + 1}/{n}  elapsed={elapsed:.1f}s  eta={eta:.1f}s')
    writer.close()
    renderer_3d.delete()
    latent_panel.close()
    print(f'wrote {args.out_mp4} ({n} frames, {n / args.fps:.1f}s @ {args.fps}fps)')
if __name__ == '__main__':
    main()
