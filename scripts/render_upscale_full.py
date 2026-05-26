import argparse
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
DEFAULT_STAGE_FRAMES = {'A': 300, 'B': 360, 'C': 1440, 'D': 360}

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

def points_to_sphere_mesh(points, radius=0.012, color=(0.91, 0.55, 0.31)):
    base = trimesh.creation.icosphere(subdivisions=1, radius=radius)
    verts_b = np.asarray(base.vertices)
    faces_b = np.asarray(base.faces)
    n_pts = len(points)
    nv = verts_b.shape[0]
    all_verts = np.empty((n_pts * nv, 3), dtype=np.float32)
    all_faces = np.empty((n_pts * faces_b.shape[0], 3), dtype=np.int32)
    for i, p in enumerate(points):
        all_verts[i * nv:(i + 1) * nv] = verts_b + p
        all_faces[i * faces_b.shape[0]:(i + 1) * faces_b.shape[0]] = faces_b + i * nv
    mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=False)
    mesh.visual.vertex_colors = (np.array([*color, 1.0]) * 255).astype(np.uint8)
    return mesh

def build_scene(panel_w, panel_h):
    scene = pyrender.Scene(bg_color=[0.04, 0.06, 0.1, 1.0], ambient_light=[0.27, 0.27, 0.29])
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=4.0), pose=look_at([2.0, 2.0, 2.5], [0, 0, 0], [0, 0, 1]))
    scene.add(pyrender.DirectionalLight(color=np.ones(3) * 0.85, intensity=2.0), pose=look_at([-2.5, -1.0, 1.5], [0, 0, 0], [0, 0, 1]))
    scene.add(pyrender.DirectionalLight(color=np.array([0.7, 0.85, 1.0]), intensity=2.5), pose=look_at([0.0, -2.5, 1.5], [0, 0, 0], [0, 0, 1]))
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=panel_w / panel_h)
    cam_node = scene.add(cam, pose=np.eye(4))
    renderer = pyrender.OffscreenRenderer(panel_w, panel_h)
    return (scene, cam_node, renderer)

def render_meshes(scene, renderer, items):
    nodes = []
    for mesh, material in items:
        if mesh is None:
            continue
        if material is None:
            pmesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        else:
            pmesh = pyrender.Mesh.from_trimesh(mesh, smooth=True, material=material)
        nodes.append(scene.add(pmesh))
    color, _ = renderer.render(scene)
    for n in nodes:
        scene.remove_node(n)
    return color

def fit_pca_2d(points):
    mean = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - mean, full_matrices=False)
    return (mean, Vt[:2])

def project(points, mean, basis):
    return (points - mean) @ basis.T

class LatentPanel:

    def __init__(self, population, z_init, z_converged, z_path, width=960, height=1080):
        self.width, self.height = (width, height)
        self.z_init = z_init
        self.z_converged = z_converged
        self.z_path = z_path
        self.mean, self.basis = fit_pca_2d(population)
        self.pop_2d = project(population, self.mean, self.basis)
        self.init_2d = project(z_init[None, :], self.mean, self.basis)[0]
        self.conv_2d = project(z_converged[None, :], self.mean, self.basis)[0]
        self.path_2d = project(z_path, self.mean, self.basis)
        all_pts = np.vstack([self.pop_2d, self.path_2d, self.init_2d[None, :]])
        xy_min, xy_max = (all_pts.min(axis=0), all_pts.max(axis=0))
        pad = 0.12 * (xy_max - xy_min).max()
        self.xlim = (xy_min[0] - pad, xy_max[0] + pad)
        self.ylim = (xy_min[1] - pad - 0.04, xy_max[1] + pad)
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
        ax.scatter(self.pop_2d[:, 0], self.pop_2d[:, 1], s=18, c='#6A7C95', alpha=0.45, linewidths=0, zorder=2)
        oy = (self.ylim[1] - self.ylim[0]) * 0.04
        self.init_dot = ax.scatter([self.init_2d[0]], [self.init_2d[1]], s=120, c='#A0A8B0', edgecolors='white', linewidths=1.4, zorder=4)
        self.init_label = ax.annotate('warm start (mean z)', xy=self.init_2d, xytext=(self.init_2d[0], self.init_2d[1] - oy), ha='center', va='top', color='#A0A8B0', fontsize=11, zorder=5)
        self.init_dot.set_visible(False)
        self.init_label.set_visible(False)
        self.conv_marker = ax.scatter([self.conv_2d[0]], [self.conv_2d[1]], s=220, facecolors='none', edgecolors='#2EAA6E', linewidths=2.4, zorder=4)
        self.conv_label = ax.annotate('upscaled ẑ', xy=self.conv_2d, xytext=(self.conv_2d[0], self.conv_2d[1] + oy), ha='center', va='bottom', color='#2EAA6E', fontsize=12, fontweight='bold', zorder=5)
        self.conv_marker.set_visible(False)
        self.conv_label.set_visible(False)
        self.path_line, = ax.plot([], [], color='#FFFFFF', linewidth=1.6, alpha=0.55, zorder=6)
        self.dot = ax.scatter([], [], s=170, c='white', edgecolors='#1B2330', linewidths=2.0, zorder=7)
        self.dot.set_visible(False)
        ax.set_xlabel('PC1', color='#A0A8B0', fontsize=11)
        ax.set_ylabel('PC2', color='#A0A8B0', fontsize=11)
        d = self.z_init.shape[0]
        ax.set_title(f'Upscaling search\n(2D PCA of 209 trained chairs, {d}-D each)', color='#D8DEE6', fontsize=13, pad=12)

    def render(self, stage, t_in_stage, frames_in_stage):
        if stage == 'A':
            self.dot.set_visible(False)
            self.path_line.set_data([], [])
            self.init_dot.set_visible(False)
            self.init_label.set_visible(False)
        elif stage == 'B':
            self.path_line.set_data([], [])
            u = t_in_stage / max(frames_in_stage - 1, 1)
            visible = u >= 0.85
            self.init_dot.set_visible(visible)
            self.init_label.set_visible(visible)
            if visible:
                self.dot.set_offsets(self.path_2d[0:1])
                self.dot.set_visible(True)
            else:
                self.dot.set_visible(False)
        elif stage == 'C':
            self.init_dot.set_visible(True)
            self.init_label.set_visible(True)
            idx = min(int(t_in_stage), len(self.path_2d) - 1)
            self.dot.set_offsets(self.path_2d[idx:idx + 1])
            self.dot.set_visible(True)
            self.path_line.set_data(self.path_2d[:idx + 1, 0], self.path_2d[:idx + 1, 1])
            if idx >= int(len(self.path_2d) * 0.75):
                self.conv_marker.set_visible(True)
                self.conv_label.set_visible(True)
        elif stage == 'D':
            self.init_dot.set_visible(True)
            self.init_label.set_visible(True)
            self.dot.set_offsets(self.path_2d[-1:])
            self.dot.set_visible(True)
            self.path_line.set_data(self.path_2d[:, 0], self.path_2d[:, 1])
            self.conv_marker.set_visible(True)
            self.conv_label.set_visible(True)
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    def close(self):
        plt.close(self.fig)

def render_stage_A(scene, renderer, input_mesh, input_material, panel, set_camera, t_offset, n_frames, hold=0):
    frames = []
    for t in range(n_frames + hold):
        eff_t = min(t, n_frames - 1)
        set_camera(t_offset + t)
        shape_img = render_meshes(scene, renderer, [(input_mesh, input_material)])
        latent_img = panel.render('A', eff_t, n_frames)
        frames.append(np.hstack([latent_img, shape_img]))
    return frames

def render_stage_B(scene, renderer, input_mesh, input_material, warm_start_mesh, warm_start_material, pts_mesh, panel, set_camera, t_offset, n_frames, hold=0):
    frames = []
    bg = np.array([10, 16, 26], dtype=np.float32)
    for t in range(n_frames + hold):
        eff_t = min(t, n_frames - 1)
        set_camera(t_offset + t)
        u = eff_t / max(n_frames - 1, 1)
        if u < 0.85:
            alpha_input = float(np.clip(1.0 - (u - 0.2) / 0.3, 0.0, 1.0))
            alpha_pts = float(np.clip(u / 0.3, 0.0, 1.0))
            input_img = render_meshes(scene, renderer, [(input_mesh, input_material)])
            pts_img = render_meshes(scene, renderer, [(pts_mesh, None)])
            out = bg + alpha_input * (input_img.astype(np.float32) - bg) + alpha_pts * (pts_img.astype(np.float32) - bg)
            out = np.clip(out, 0, 255).astype(np.uint8)
        else:
            out = render_meshes(scene, renderer, [(warm_start_mesh, warm_start_material), (pts_mesh, None)])
        latent_img = panel.render('B', eff_t, n_frames)
        frames.append(np.hstack([latent_img, out]))
    return frames

def render_stage_C(scene, renderer, completion_objs, completion_material, pts_mesh, panel, set_camera, t_offset, n_frames, hold=0, start_hold=0):
    frames = []
    n_meshes = len(completion_objs)
    last_idx = -1
    comp_mesh = None
    total = start_hold + n_frames + hold
    for t in range(total):
        if t < start_hold:
            morph_t = 0
        else:
            morph_t = min(t - start_hold, n_frames - 1)
        set_camera(t_offset + t)
        mesh_idx = min(int(morph_t * n_meshes / max(n_frames, 1)), n_meshes - 1)
        if mesh_idx != last_idx:
            comp_mesh = trimesh.load(completion_objs[mesh_idx], process=False)
            last_idx = mesh_idx
        shape_img = render_meshes(scene, renderer, [(comp_mesh, completion_material), (pts_mesh, None)])
        latent_img = panel.render('C', mesh_idx, n_meshes)
        frames.append(np.hstack([latent_img, shape_img]))
    return frames

def render_stage_D(scene, renderer, final_comp_mesh, completion_material, input_mesh, input_material, pts_mesh, panel, set_camera, t_offset, n_frames, hold=0):
    frames = []
    bg = np.array([10, 16, 26], dtype=np.float32)
    for t in range(n_frames + hold):
        eff_t = min(t, n_frames - 1)
        set_camera(t_offset + t)
        u = eff_t / max(n_frames - 1, 1)
        comp_img = render_meshes(scene, renderer, [(final_comp_mesh, completion_material), (pts_mesh, None)])
        alpha_in = float(np.clip((u - 0.3) / 0.6, 0.0, 1.0)) * 0.6
        if alpha_in > 0:
            in_img = render_meshes(scene, renderer, [(input_mesh, input_material)])
            mixed = comp_img.astype(np.float32) + alpha_in * (in_img.astype(np.float32) - bg)
            shape_img = np.clip(mixed, 0, 255).astype(np.uint8)
        else:
            shape_img = comp_img
        latent_img = panel.render('D', eff_t, n_frames)
        frames.append(np.hstack([latent_img, shape_img]))
    return frames

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meshes-dir', required=True, help='upscale_bake output: frame_NNNN.obj + z_path.npy')
    ap.add_argument('--traj-dir', required=True, help='upscale.py output: input_lowres.obj, partial_points.npy, partial_sdf.npy')
    ap.add_argument('--all-latents', required=True, help='results.npy with the 209 trained latents')
    ap.add_argument('--out-mp4', default='upscale_full.mp4')
    ap.add_argument('--out-png-dir', default='upscale_full_png')
    ap.add_argument('--panel-width', type=int, default=960)
    ap.add_argument('--panel-height', type=int, default=1080)
    ap.add_argument('--fps', type=int, default=120)
    ap.add_argument('--cam-distance', type=float, default=1.7)
    ap.add_argument('--cam-height', type=float, default=0.35)
    ap.add_argument('--start-angle-deg', type=float, default=0.0)
    ap.add_argument('--rotation-deg', type=float, default=180.0)
    ap.add_argument('--surface-sdf-thresh', type=float, default=0.001)
    ap.add_argument('--max-points', type=int, default=600)
    ap.add_argument('--inter-stage-hold', type=int, default=240)
    ap.add_argument('--c-start-hold', type=int, default=360)
    args = ap.parse_args()
    meshes_dir = Path(args.meshes_dir)
    traj_dir = Path(args.traj_dir)
    out_png_dir = Path(args.out_png_dir)
    out_png_dir.mkdir(parents=True, exist_ok=True)
    input_mesh = trimesh.load(traj_dir / 'input_lowres.obj', process=False)
    z_path = np.load(meshes_dir / 'z_path.npy')
    partial_pts_all = np.load(traj_dir / 'partial_points.npy')
    partial_sdf = np.load(traj_dir / 'partial_sdf.npy')
    population = np.load(args.all_latents, allow_pickle=True).item()['best_latent_codes']
    z_init = population.mean(axis=0).astype(np.float32)
    surface_mask = np.abs(partial_sdf) < args.surface_sdf_thresh
    surface_pts = partial_pts_all[surface_mask]
    print(f'surface filter |sdf|<{args.surface_sdf_thresh}: {len(surface_pts)}/{len(partial_pts_all)} points kept')
    if len(surface_pts) > args.max_points:
        rng = np.random.RandomState(0)
        sel = rng.choice(len(surface_pts), args.max_points, replace=False)
        surface_pts = surface_pts[sel]
    pts_mesh = points_to_sphere_mesh(surface_pts, radius=0.012)
    completion_objs = sorted(meshes_dir.glob('frame_*.obj'))
    z_converged = z_path[-1].astype(np.float32)
    final_comp_mesh = trimesh.load(completion_objs[-1], process=False)
    scene, cam_node, renderer = build_scene(args.panel_width, args.panel_height)
    input_material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.93, 0.85, 0.66, 1.0], metallicFactor=0.0, roughnessFactor=0.65)
    completion_material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.78, 0.86, 0.95, 1.0], metallicFactor=0.0, roughnessFactor=0.55)
    panel = LatentPanel(population=population, z_init=z_init, z_converged=z_converged, z_path=z_path, width=args.panel_width, height=args.panel_height)
    H = args.inter_stage_hold
    stage_holds = {'A': H, 'B': H, 'C': H, 'D': 0}
    stages = [('A', DEFAULT_STAGE_FRAMES['A']), ('B', DEFAULT_STAGE_FRAMES['B']), ('C', DEFAULT_STAGE_FRAMES['C']), ('D', DEFAULT_STAGE_FRAMES['D'])]
    n_total = sum((n + stage_holds[name] for name, n in stages)) + args.c_start_hold
    print(f'rendering {n_total} frames @ {2 * args.panel_width}x{args.panel_height}, fps={args.fps}, total={n_total / args.fps:.1f}s (inter-stage hold = {H}, C start-hold = {args.c_start_hold})')
    rot_total = np.radians(args.rotation_deg)
    start_offset = np.radians(args.start_angle_deg)

    def set_camera(global_idx):
        theta = start_offset - rot_total / 2 + rot_total * (global_idx / max(n_total - 1, 1))
        eye = np.array([args.cam_distance * np.sin(theta), -args.cam_distance * np.cos(theta), args.cam_height], dtype=np.float32)
        scene.set_pose(cam_node, look_at(eye, [0, 0, 0], [0, 0, 1]))
    writer = imageio.get_writer(args.out_mp4, fps=args.fps, codec='libx264', quality=7, macro_block_size=1)
    t_start = time.time()
    t_offset = 0
    for stage, n_frames in stages:
        h = stage_holds[stage]
        c_start = args.c_start_hold if stage == 'C' else 0
        total_in_stage = n_frames + h + c_start
        print(f'  stage {stage}: {n_frames} (+{c_start} pre, +{h} post) frames')
        if stage == 'A':
            frames = render_stage_A(scene, renderer, input_mesh, input_material, panel, set_camera, t_offset, n_frames, hold=h)
        elif stage == 'B':
            warm_start_mesh = trimesh.load(completion_objs[0], process=False)
            frames = render_stage_B(scene, renderer, input_mesh, input_material, warm_start_mesh, completion_material, pts_mesh, panel, set_camera, t_offset, n_frames, hold=h)
        elif stage == 'C':
            frames = render_stage_C(scene, renderer, completion_objs, completion_material, pts_mesh, panel, set_camera, t_offset, n_frames, hold=h, start_hold=c_start)
        elif stage == 'D':
            frames = render_stage_D(scene, renderer, final_comp_mesh, completion_material, input_mesh, input_material, pts_mesh, panel, set_camera, t_offset, n_frames, hold=h)
        for i, frame in enumerate(frames):
            imageio.imwrite(out_png_dir / f'frame_{t_offset + i:04d}.png', frame)
            writer.append_data(frame)
        t_offset += total_in_stage
        elapsed = time.time() - t_start
        print(f'    done; total elapsed = {elapsed:.0f}s')
    writer.close()
    renderer.delete()
    panel.close()
    print(f'wrote {args.out_mp4} ({n_total} frames, {n_total / args.fps:.1f}s @ {args.fps}fps)')
if __name__ == '__main__':
    main()
