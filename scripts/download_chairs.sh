#!/bin/bash
# Download the 209 ABO chair GLBs listed in CSV.
# CSV is expected to have an item_id column (any position), typically
# data/abo_chairs_used.csv with columns (latent_idx, item_id, product_type).
# S3 paths are looked up from ABO's published 3dmodels.csv metadata, which
# this script downloads on first run.
set -e

CSV="$1"
OUTDIR="$2"
META_DIR="${3:-data/abo_meta}"

if [ -z "$CSV" ] || [ -z "$OUTDIR" ]; then
  echo "usage: $0 <chairs.csv> <out-dir> [meta-dir]" >&2
  exit 1
fi
mkdir -p "$OUTDIR" "$META_DIR"

META_CSV="$META_DIR/3dmodels.csv"
if [ ! -s "$META_CSV" ]; then
  echo "fetching ABO metadata 3dmodels.csv -> $META_CSV"
  curl -sLf -o "$META_DIR/3dmodels.csv.gz" \
    https://amazon-berkeley-objects.s3.amazonaws.com/3dmodels/metadata/3dmodels.csv.gz
  gunzip -f "$META_DIR/3dmodels.csv.gz"
fi

# Build id -> path table from ABO metadata. ABO 3dmodels.csv columns:
# 3dmodel_id,path,meshes,materials,...   path is the S3 relative GLB key.
ID_COL=$(head -n 1 "$META_CSV" | tr ',' '\n' | grep -n '^3dmodel_id$' | cut -d: -f1)
PATH_COL=$(head -n 1 "$META_CSV" | tr ',' '\n' | grep -n '^path$' | cut -d: -f1)
if [ -z "$ID_COL" ] || [ -z "$PATH_COL" ]; then
  echo "could not locate 3dmodel_id / path columns in $META_CSV" >&2
  exit 1
fi

# Build the same lookup for the input CSV
IN_ID_COL=$(head -n 1 "$CSV" | tr ',' '\n' | grep -n '^item_id$' | cut -d: -f1)
if [ -z "$IN_ID_COL" ]; then
  echo "input CSV $CSV needs an item_id column" >&2
  exit 1
fi

awk -F, -v id_col="$ID_COL" -v path_col="$PATH_COL" \
       -v in_id_col="$IN_ID_COL" -v outdir="$OUTDIR" \
       -v csv="$CSV" -v meta="$META_CSV" '
  FNR == NR { if (FNR > 1) path[$id_col] = $path_col; next }
  FNR == 1 { next }
  { id = $in_id_col;
    if (id in path)
      printf "https://amazon-berkeley-objects.s3.amazonaws.com/3dmodels/original/%s\t%s/%s.glb\n",
             path[id], outdir, id;
    else
      print "MISSING " id > "/dev/stderr" }
' "$META_CSV" "$CSV" | while IFS=$'\t' read -r url out; do
  [ -s "$out" ] && continue
  printf "%s\n%s\n" "$url" "$out"
done | xargs -P 4 -n 2 bash -c '
  if curl -sLf -o "$2" "$1"; then
    echo "ok $(stat -c%s "$2") $2"
  else
    echo "FAIL $2"
  fi
' _
