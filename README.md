# DeepSDF Demo
 
A demo of the DeepSDF auto-decoder trained on 209 chair-like shapes from the Amazon Berkeley Objects 3D collection. Reproduces the animations from the presentation at https://slides.adriansteffan.com/deepsdf as well as the renders from the report (`Report_on_DeepSDF.pdf`, included in this repo). This project builds on a third-party reimplementation of DeepSDF at https://github.com/maurock/DeepSDF (commit `a645549`). The adaptation and implementation of the demos was assisted by Claude Code Opus 4.7.


## Reproducing

Requires Python 3.12 + a CUDA GPU.

Steps 1-4 (download / convert / sample / train) are only needed to rebuild the checkpoint from scratch.
Skip to step 5 to render videos (or to the next section for the figure strips) from what already ships in `model/`.

```bash
python3.12 -m venv .venv
.venv/bin/pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -r requirements.txt

# 1. Download 209 ABO chair GLBs (~10 GB)
bash scripts/download_chairs.sh data/abo_chairs_used.csv data/glb

# 2. Convert ABO objects to ShapeNetCore-like layout
.venv/bin/python scripts/abo_to_obj.py --csv data/abo_chairs_used.csv \
    --glb-root data/glb --out-root data/ShapeNetCoreV2

# 3. SDF samples
.venv/bin/python scripts/extract_sdf_abo.py \
    --data-root data/ShapeNetCoreV2 --results-dir model

# 4. Train
.venv/bin/python scripts/train_sdf.py \
    --results-dir model --inner-dim 512 --latent-size 256 \
    --epochs 1000 --batch-size 4096 --tag chairs

# RUN defaults to the shipped checkpoint. If you ran step 4 yourself, point
# it at the timestamped folder in model/runs_sdf/ instead. Note that
# retraining permutes latent ordering, so 5a's `--latent-indices` and 5b's
# `--target-idx` (both picked against the shipped checkpoint) will land on
# different chairs. Use `render_all_anchors.py` to reselect anchors in that
# case.
RUN=model/run_shipped

# 5a. Morph
.venv/bin/python scripts/interpolation_script.py --run-dir $RUN \
    --latent-indices 51,111,120 --num-frames 600 --resolution 256 \
    --out-dir morph_frames
.venv/bin/python scripts/render_morph.py --frames-dir morph_frames \
    --all-latents $RUN/results.npy --labels "bar stool,club chair,high-back chair" \
    --out-mp4 videos/morph.mp4 --fps 60 --loop --start-angle-deg 90 \
    --panel-order latent-shape

# 5b. Completion (target chair idx 16 = wing chair B075X4PTSN)
.venv/bin/python scripts/completion_script.py --run-dir $RUN \
    --samples-npy model/samples_dict_chair16.npy \
    --target-idx 16 --mask "z<0" --steps 400 --reg-coef 50 \
    --out-dir completion_traj
.venv/bin/python scripts/completion_bake.py --run-dir $RUN \
    --traj-dir completion_traj --out-dir completion_meshes \
    --n-frames 540 --resolution 256
.venv/bin/python scripts/render_completion.py \
    --meshes-dir completion_meshes --traj-dir completion_traj \
    --all-latents $RUN/results.npy --out-mp4 videos/completion.mp4 \
    --fps 120 --start-angle-deg 90

# 5c. Upscale (target chair idx 197 = B0853NR959 tub chair)
OBJ=model/input_lowres.obj
.venv/bin/python scripts/upscale.py --run-dir $RUN --obj-path $OBJ \
    --target-faces 80 --steps 500 --reg-coef 120 --out-dir upscale_traj
.venv/bin/python scripts/upscale_bake.py --run-dir $RUN \
    --traj-dir upscale_traj --out-dir upscale_meshes \
    --n-frames 540 --resolution 256
.venv/bin/python scripts/render_upscale_full.py \
    --meshes-dir upscale_meshes --traj-dir upscale_traj \
    --all-latents $RUN/results.npy --out-mp4 videos/upscale.mp4 \
    --fps 120 --rotation-deg 450 --start-angle-deg 225
```

## Regenerating the paper figures

The strips in `figures/` are produced by two scripts that operate against
the shipped checkpoint, no retraining needed:

```bash
# morph strip (club chair -> high-back, 7 frames)
.venv/bin/python scripts/render_steps.py \
    --from-idx 111 --to-idx 120 --out-dir figures/morph_steps

# upscale strip (low-res input -> 6 latent steps to upscaled mesh)
.venv/bin/python scripts/render_upscale_steps.py \
    --out-dir figures/upscale_steps
```

Both write per-frame PNGs and a stitched `strip.png`.
Use `--help` on each for camera / aspect controls.

## License

- ABO meshes: CC BY 4.0 (https://amazon-berkeley-objects.s3.amazonaws.com/).
- Files adapted from the maurock/DeepSDF implementation (`scripts/sdf_model.py`,
  `scripts/train_sdf.py`, parts of `scripts/extract_sdf_abo.py`) inherit
  that repo's MIT license.

