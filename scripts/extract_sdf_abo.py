import argparse
import time
from glob import glob
from pathlib import Path
import numpy as np
import point_cloud_utils as pcu
import trimesh

def as_single_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        return trimesh.util.concatenate([trimesh.Trimesh(vertices=m.vertices, faces=m.faces) for m in scene_or_mesh.geometry.values()])
    return scene_or_mesh

def sample_one(obj_path, n_vol, n_bbox, n_surf, watertight_target=50000):
    mesh_t = as_single_mesh(trimesh.load(obj_path, process=False))
    verts = np.asarray(mesh_t.vertices, dtype=np.float64)
    faces = np.asarray(mesh_t.faces, dtype=np.int32)
    if not mesh_t.is_watertight:
        verts, faces = pcu.make_mesh_watertight(verts, faces, watertight_target)
    p_vol = np.random.rand(n_vol, 3).astype(np.float32) * 2.0 - 1.0
    v_min = verts.min(0).astype(np.float32)
    v_max = verts.max(0).astype(np.float32)
    p_bbox = np.random.uniform(low=v_min, high=v_max, size=(n_bbox, 3)).astype(np.float32)
    fid_surf, bc_surf = pcu.sample_mesh_random(verts.astype(np.float32), faces.astype(np.int32), n_surf)
    p_surf = pcu.interpolate_barycentric_coords(faces.astype(np.int32), fid_surf, bc_surf, verts.astype(np.float32))
    p_total = np.vstack((p_vol, p_bbox, p_surf)).astype(np.float32)
    sdf, _, _ = pcu.signed_distance_to_mesh(p_total, verts.astype(np.float32), faces.astype(np.int32))
    return (p_total, sdf.astype(np.float32))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='data/ShapeNetCoreV2')
    ap.add_argument('--synset', default='03001627')
    ap.add_argument('--results-dir', default='model')
    ap.add_argument('--dataset-name', default='ShapeNetCore')
    ap.add_argument('--n-vol', type=int, default=3000)
    ap.add_argument('--n-bbox', type=int, default=10000)
    ap.add_argument('--n-surf', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    np.random.seed(args.seed)
    obj_paths = sorted(glob(str(Path(args.data_root) / args.synset / '*' / 'models' / '*.obj')))
    print(f'{len(obj_paths)} OBJs under synset {args.synset}')
    samples_dict, str2int, int2str = {}, {}, {}
    t0 = time.time()
    for obj_idx, obj_path in enumerate(obj_paths):
        parts = Path(obj_path).parts
        key = f'{parts[-4]}/{parts[-3]}'
        str2int[key] = obj_idx
        int2str[obj_idx] = key
        try:
            t1 = time.time()
            p, sdf = sample_one(obj_path, args.n_vol, args.n_bbox, args.n_surf)
            dt = time.time() - t1
        except Exception as e:
            print(f'  [{obj_idx:3d}] {key}  FAILED: {type(e).__name__}: {e}')
            continue
        samples_dict[obj_idx] = {
            'sdf': sdf,
            'samples_latent_class': np.hstack([np.full((p.shape[0], 1), obj_idx, dtype=np.int32), p.astype(np.float32)]),
        }
        print(f'  [{obj_idx:3d}] {key}  pts={p.shape[0]}  in={int((sdf<0).sum())}  out={int((sdf>0).sum())}  ({dt:.1f}s)')
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    np.save(Path(args.results_dir) / f'samples_dict_{args.dataset_name}.npy', samples_dict, allow_pickle=True)
    np.save(Path(args.results_dir) / 'idx_str2int_dict.npy', str2int, allow_pickle=True)
    np.save(Path(args.results_dir) / 'idx_int2str_dict.npy', int2str, allow_pickle=True)
    print(f'{len(samples_dict)} shapes in {time.time() - t0:.1f}s')
if __name__ == '__main__':
    main()
