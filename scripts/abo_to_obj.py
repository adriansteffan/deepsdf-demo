import argparse
import csv
from pathlib import Path
import numpy as np
import trimesh

CHAIR_SYNSET = '03001627'

def load_glb_as_single_mesh(glb_path):
    scene_or_mesh = trimesh.load(glb_path, process=False, force='scene')
    if isinstance(scene_or_mesh, trimesh.Scene):
        geoms = []
        for name, geom in scene_or_mesh.geometry.items():
            if not isinstance(geom, trimesh.Trimesh):
                continue
            for node_name in scene_or_mesh.graph.nodes_geometry:
                if scene_or_mesh.graph[node_name][1] == name:
                    T = scene_or_mesh.graph[node_name][0]
                    g = geom.copy()
                    g.apply_transform(T)
                    geoms.append(g)
                    break
            else:
                geoms.append(geom.copy())
        if not geoms:
            raise ValueError(f'no Trimesh geometry in {glb_path}')
        mesh = trimesh.util.concatenate(geoms)
    else:
        mesh = scene_or_mesh
    return mesh

def y_up_to_z_up(mesh):
    R = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    mesh.apply_transform(R)
    return mesh

def normalize_unit_bbox(mesh):
    bounds = mesh.bounds
    centroid = (bounds[0] + bounds[1]) / 2.0
    mesh.apply_translation(-centroid)
    extent = (bounds[1] - bounds[0]).max()
    if extent <= 0:
        raise ValueError('zero-extent mesh')
    mesh.apply_scale(1.0 / extent)
    return mesh

def convert_one(glb_path, item_id, out_root):
    out_dir = Path(out_root) / CHAIR_SYNSET / item_id / 'models'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'model_normalized.obj'
    mesh = load_glb_as_single_mesh(glb_path)
    mesh = y_up_to_z_up(mesh)
    mesh = normalize_unit_bbox(mesh)
    out_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    out_mesh.export(out_path, file_type='obj')
    return (out_path, len(mesh.vertices), len(mesh.faces))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--glb-root', required=True)
    ap.add_argument('--out-root', default='data/ShapeNetCoreV2')
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.csv)))
    print(f'converting {len(rows)} GLBs from {args.glb_root}')
    for i, row in enumerate(rows, 1):
        item_id = row['item_id']
        glb = Path(args.glb_root) / f'{item_id}.glb'
        if not glb.exists():
            print(f'  {i:3d}/{len(rows)}  {item_id}  MISSING')
            continue
        try:
            _, nv, nf = convert_one(glb, item_id, args.out_root)
            print(f'  {i:3d}/{len(rows)}  {item_id}  V={nv:>7d} F={nf:>7d}')
        except Exception as e:
            print(f'  {i:3d}/{len(rows)}  {item_id}  FAILED: {type(e).__name__}: {e}')
if __name__ == '__main__':
    main()
