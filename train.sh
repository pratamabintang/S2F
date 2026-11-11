#!/usr/bin/env bash
# =============================================================================
# Skrip Pelatihan Few-Shot Segmentation (S2F) - Multimodal Landslide
# =============================================================================
# Satu eksekusi = satu konfigurasi. Semua parameter bisa diatur lewat flag CLI
# atau variabel environment.
#
# Resolusi path default mengikuti struktur Google Drive:
#   TRAIN_ROOT = /content/drive/MyDrive/project/3.TRAINING
#   VAL_ROOT   = /content/drive/MyDrive/project/4.VALIDATION
#   TEST_ROOT  = /content/drive/MyDrive/project/5.TESTING
#   blacklist  = /content/drive/MyDrive/project/black_list.txt
#   checkpoint = /content/drive/MyDrive/project/checkpoint/sam2_hiera_large.pt
#   runs       = /content/drive/MyDrive/project/runs
# Jika Drive tidak tersedia, otomatis fallback ke folder lokal repositori.
#
# Contoh:
#   ./train.sh --feature dtm --shot 1
#   ./train.sh --topo DTM HILLSHADE --shot 5 --epochs 30 --batch-size 2 --test
#   ./train.sh --train-root /data/landslide/train --val-root /data/landslide/val
# =============================================================================

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
REPO_DIR="${REPO_DIR:-$(pwd)}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
export OPENCV_LOG_LEVEL="SILENT"
export PYTHONWARNINGS="ignore"

# -----------------------------------------------------------------------------
# Default (bisa dioverride lewat environment)
# -----------------------------------------------------------------------------
DATASET="${DATASET:-landslide}"

# Path split dataset (default Google Drive, fallback lokal otomatis)
DATA_ROOT="${DATA_ROOT:-${DATASET_ROOT:-}}"
TRAIN_ROOT="${TRAIN_ROOT:-/content/drive/MyDrive/project/3.TRAINING}"
VAL_ROOT="${VAL_ROOT:-/content/drive/MyDrive/project/4.VALIDATION}"
TEST_ROOT="${TEST_ROOT:-/content/drive/MyDrive/project/5.TESTING}"
BLACKLIST="${BLACKLIST:-${BLACK_LIST_PATH:-}}"
HIERA_PATH="${HIERA_PATH:-${HIERA_LARGE_PATH:-/content/drive/MyDrive/project/checkpoint/sam2_hiera_large.pt}}"
OUTPUT_DIR="${OUTPUT_DIR:-/content/drive/MyDrive/project/runs}"

FEATURE="${FEATURE:-dtm}"
TOPO_OVERRIDE="${TOPO_OVERRIDE:-}"

SHOT="${SHOT:-1}"
SIZE="${SIZE:-352}"
REDUCE_DIM="${REDUCE_DIM:-64}"
EPOCHS="${EPOCHS:-20}"
EPISODES_PER_EPOCH="${EPISODES_PER_EPOCH:-1000}"
VAL_EPISODES="${VAL_EPISODES:-1000}"
VAL_INTERVAL="${VAL_INTERVAL:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-0}"
AMP_DTYPE="${AMP_DTYPE:-auto}"
CACHE_DIR="${CACHE_DIR:-}"

RUN_NAME="${RUN_NAME:-}"
SAVE_PATH="${SAVE_PATH:-}"
FOLD="${FOLD:-0}"
RUN_TEST="false"
TEST_CHECKPOINT="${TEST_CHECKPOINT:-}"

# PASCAL paths
P_TRAIN_IMG_DIR="${P_TRAIN_IMG_DIR:-}"
P_TRAIN_MASK_DIR="${P_TRAIN_MASK_DIR:-}"
P_TRAIN_SPLIT="${P_TRAIN_SPLIT:-}"
P_VAL_IMG_DIR="${P_VAL_IMG_DIR:-}"
P_VAL_MASK_DIR="${P_VAL_MASK_DIR:-}"
P_VAL_SPLIT="${P_VAL_SPLIT:-}"
P_TEST_IMG_DIR="${P_TEST_IMG_DIR:-}"
P_TEST_MASK_DIR="${P_TEST_MASK_DIR:-}"
P_TEST_SPLIT="${P_TEST_SPLIT:-}"

show_help() {
    cat <<'EOF'
=============================================================================
S2F Few-Shot Training (multimodal landslide / PASCAL-5i)
=============================================================================
Penggunaan:
  ./train.sh [OPSI...]

Dataset (default path Google Drive, fallback lokal otomatis):
  --dataset {landslide|pascal}   default: landslide
  --train-root PATH              root split train (default: .../3.TRAINING)
  --val-root PATH                root split val   (default: .../4.VALIDATION)
  --test-root PATH               root split test  (default: .../5.TESTING)
  --data-root PATH               root induk opsional; auto-deteksi 3.TRAINING/
                                 4.VALIDATION/5.TESTING atau train/val/test
  --blacklist PATH|none          default: /content/drive/MyDrive/project/black_list.txt

Topografi (landslide):
  --feature NAME                 dtm|slope|hillshade|aspect|aspectsin|aspectcos|kombinasi
  --topo MOD [MOD ...]           eksplisit, mis: --topo DTM HILLSHADE

Model & episode:
  --shot INT                     jumlah support (default 1)
  --size INT                     resolusi input (default 352)
  --reduce-dim INT               dimensi reduce (default 64)
  --hiera-path PATH              checkpoint SAM2
                                 (default: .../checkpoint/sam2_hiera_large.pt)

Training:
  --epochs INT                   default 20
  --episodes-per-epoch INT       default 1000
  --val-episodes INT             default 1000
  --val-interval INT             default 1
  --batch-size INT               default 4
  --lr FLOAT                     default 1e-3
  --weight-decay FLOAT           default 5e-4
  --num-workers INT              default 4
  --seed INT                     default 0
  --amp-dtype {auto|bf16|fp16|none}
  --cache-dir PATH               cache index positif

Output:
  --output PATH                  folder runs (default: .../project/runs)
  --run-name NAME                nama run (default: auto)
  --save-path PATH               override folder checkpoint save
  --test                         jalankan test_fss.py setelah training
  --test-checkpoint PATH         checkpoint untuk test (default: best)

PASCAL:
  --fold N --train-img-dir D --train-mask-dir D --train-split F
  --val-img-dir D --val-mask-dir D --val-split F
  --test-img-dir D --test-mask-dir D --test-split F

Contoh:
  ./train.sh --topo DTM HILLSHADE --shot 5 --test
  ./train.sh --train-root /data/landslide/train --val-root /data/landslide/val
=============================================================================
EOF
}

# -----------------------------------------------------------------------------
# Parsing argumen
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    arg="$1"
    case "$arg" in
        -h|--help)
            show_help; exit 0 ;;
        --dataset|--dataset=*)
            if [[ "$arg" == *=* ]]; then DATASET="${arg#*=}"; else DATASET="$2"; shift; fi ;;
        --data-root|--data_root|--data-root=*|--data_root=*)
            if [[ "$arg" == *=* ]]; then DATA_ROOT="${arg#*=}"; else DATA_ROOT="$2"; shift; fi ;;
        --train-root|--train_root|--train-root=*|--train_root=*|--train-dir|--train_dir)
            if [[ "$arg" == *=* ]]; then TRAIN_ROOT="${arg#*=}"; else TRAIN_ROOT="$2"; shift; fi ;;
        --val-root|--val_root|--val-root=*|--val_root=*|--val-dir|--val_dir)
            if [[ "$arg" == *=* ]]; then VAL_ROOT="${arg#*=}"; else VAL_ROOT="$2"; shift; fi ;;
        --test-root|--test_root|--test-root=*|--test_root=*|--test-dir|--test_dir)
            if [[ "$arg" == *=* ]]; then TEST_ROOT="${arg#*=}"; else TEST_ROOT="$2"; shift; fi ;;
        --blacklist|--blacklist=*)
            if [[ "$arg" == *=* ]]; then BLACKLIST="${arg#*=}"; else BLACKLIST="$2"; shift; fi ;;
        --feature|--feature=*|-f|-f=*)
            if [[ "$arg" == *=* ]]; then FEATURE="${arg#*=}"; else FEATURE="$2"; shift; fi ;;
        --topo|--topo=*)
            if [[ "$arg" == *=* ]]; then
                TOPO_OVERRIDE="${arg#*=}"
            else
                TOPO_OVERRIDE=""
                shift
                while [[ $# -gt 0 && "$1" != -* ]]; do
                    TOPO_OVERRIDE+=" $1"
                    shift
                done
                TOPO_OVERRIDE="${TOPO_OVERRIDE# }"
                continue
            fi ;;
        --shot|--shot=*)
            if [[ "$arg" == *=* ]]; then SHOT="${arg#*=}"; else SHOT="$2"; shift; fi ;;
        --size|--size=*)
            if [[ "$arg" == *=* ]]; then SIZE="${arg#*=}"; else SIZE="$2"; shift; fi ;;
        --reduce-dim|--reduce_dim|--reduce-dim=*|--reduce_dim=*)
            if [[ "$arg" == *=* ]]; then REDUCE_DIM="${arg#*=}"; else REDUCE_DIM="$2"; shift; fi ;;
        --epochs|--epochs=*)
            if [[ "$arg" == *=* ]]; then EPOCHS="${arg#*=}"; else EPOCHS="$2"; shift; fi ;;
        --episodes-per-epoch|--episodes_per_epoch|--episodes-per-epoch=*|--episodes_per_epoch=*)
            if [[ "$arg" == *=* ]]; then EPISODES_PER_EPOCH="${arg#*=}"; else EPISODES_PER_EPOCH="$2"; shift; fi ;;
        --val-episodes|--val_episodes|--val-episodes=*|--val_episodes=*)
            if [[ "$arg" == *=* ]]; then VAL_EPISODES="${arg#*=}"; else VAL_EPISODES="$2"; shift; fi ;;
        --val-interval|--val_interval|--val-interval=*|--val_interval=*)
            if [[ "$arg" == *=* ]]; then VAL_INTERVAL="${arg#*=}"; else VAL_INTERVAL="$2"; shift; fi ;;
        --batch-size|--batch_size|--batch-size=*|--batch_size=*)
            if [[ "$arg" == *=* ]]; then BATCH_SIZE="${arg#*=}"; else BATCH_SIZE="$2"; shift; fi ;;
        --lr|--lr=*)
            if [[ "$arg" == *=* ]]; then LR="${arg#*=}"; else LR="$2"; shift; fi ;;
        --weight-decay|--weight_decay|--weight-decay=*|--weight_decay=*)
            if [[ "$arg" == *=* ]]; then WEIGHT_DECAY="${arg#*=}"; else WEIGHT_DECAY="$2"; shift; fi ;;
        --num-workers|--num_workers|--num-workers=*|--num_workers=*)
            if [[ "$arg" == *=* ]]; then NUM_WORKERS="${arg#*=}"; else NUM_WORKERS="$2"; shift; fi ;;
        --seed|--seed=*)
            if [[ "$arg" == *=* ]]; then SEED="${arg#*=}"; else SEED="$2"; shift; fi ;;
        --amp-dtype|--amp_dtype|--amp-dtype=*|--amp_dtype=*)
            if [[ "$arg" == *=* ]]; then AMP_DTYPE="${arg#*=}"; else AMP_DTYPE="$2"; shift; fi ;;
        --cache-dir|--cache_dir|--cache-dir=*|--cache_dir=*)
            if [[ "$arg" == *=* ]]; then CACHE_DIR="${arg#*=}"; else CACHE_DIR="$2"; shift; fi ;;
        --hiera-path|--hiera_path|--hiera-path=*|--hiera_path=*|--hiera-large-path|--hiera_large_path)
            if [[ "$arg" == *=* ]]; then HIERA_PATH="${arg#*=}"; else HIERA_PATH="$2"; shift; fi ;;
        --output|--output=*|--output-dir|--output-dir=*|--output_training_path|--output-training-path)
            if [[ "$arg" == *=* ]]; then OUTPUT_DIR="${arg#*=}"; else OUTPUT_DIR="$2"; shift; fi ;;
        --run-name|--run_name|--run-name=*|--run_name=*)
            if [[ "$arg" == *=* ]]; then RUN_NAME="${arg#*=}"; else RUN_NAME="$2"; shift; fi ;;
        --save-path|--save_path|--save-path=*|--save_path=*)
            if [[ "$arg" == *=* ]]; then SAVE_PATH="${arg#*=}"; else SAVE_PATH="$2"; shift; fi ;;
        --fold|--fold=*)
            if [[ "$arg" == *=* ]]; then FOLD="${arg#*=}"; else FOLD="$2"; shift; fi ;;
        --test-checkpoint|--test_checkpoint|--test-checkpoint=*|--test_checkpoint=*)
            if [[ "$arg" == *=* ]]; then TEST_CHECKPOINT="${arg#*=}"; else TEST_CHECKPOINT="$2"; shift; fi ;;
        --test|--run-test)
            RUN_TEST="true" ;;
        --train-img-dir|--train_img_dir|--train-img-dir=*|--train_img_dir=*)
            if [[ "$arg" == *=* ]]; then P_TRAIN_IMG_DIR="${arg#*=}"; else P_TRAIN_IMG_DIR="$2"; shift; fi ;;
        --train-mask-dir|--train_mask_dir|--train-mask-dir=*|--train_mask_dir=*)
            if [[ "$arg" == *=* ]]; then P_TRAIN_MASK_DIR="${arg#*=}"; else P_TRAIN_MASK_DIR="$2"; shift; fi ;;
        --train-split|--train_split|--train-split=*|--train_split=*)
            if [[ "$arg" == *=* ]]; then P_TRAIN_SPLIT="${arg#*=}"; else P_TRAIN_SPLIT="$2"; shift; fi ;;
        --val-img-dir|--val_img_dir|--val-img-dir=*|--val_img_dir=*)
            if [[ "$arg" == *=* ]]; then P_VAL_IMG_DIR="${arg#*=}"; else P_VAL_IMG_DIR="$2"; shift; fi ;;
        --val-mask-dir|--val_mask_dir|--val-mask-dir=*|--val_mask_dir=*)
            if [[ "$arg" == *=* ]]; then P_VAL_MASK_DIR="${arg#*=}"; else P_VAL_MASK_DIR="$2"; shift; fi ;;
        --val-split|--val_split|--val-split=*|--val_split=*)
            if [[ "$arg" == *=* ]]; then P_VAL_SPLIT="${arg#*=}"; else P_VAL_SPLIT="$2"; shift; fi ;;
        --test-img-dir|--test_img_dir|--test-img-dir=*|--test_img_dir=*)
            if [[ "$arg" == *=* ]]; then P_TEST_IMG_DIR="${arg#*=}"; else P_TEST_IMG_DIR="$2"; shift; fi ;;
        --test-mask-dir|--test_mask_dir|--test-mask-dir=*|--test_mask_dir=*)
            if [[ "$arg" == *=* ]]; then P_TEST_MASK_DIR="${arg#*=}"; else P_TEST_MASK_DIR="$2"; shift; fi ;;
        --test-split|--test_split|--test-split=*|--test_split=*)
            if [[ "$arg" == *=* ]]; then P_TEST_SPLIT="${arg#*=}"; else P_TEST_SPLIT="$2"; shift; fi ;;
        *)
            echo "ERROR: argumen tidak dikenal: $arg" >&2
            echo "Gunakan --help." >&2
            exit 2 ;;
    esac
    shift
done

# -----------------------------------------------------------------------------
# Resolusi topografi
# -----------------------------------------------------------------------------
TOPO_LIST=()
if [[ -n "$TOPO_OVERRIDE" ]]; then
    TOPO_OVERRIDE="${TOPO_OVERRIDE//,/ }"
    read -ra TOPO_LIST <<< "$TOPO_OVERRIDE"
else
    case "$(echo "$FEATURE" | tr '[:upper:]' '[:lower:]' | tr -d ' -_')" in
        dtm|dem)              TOPO_LIST=(DTM) ;;
        slope)                TOPO_LIST=(SLOPE) ;;
        aspect)               TOPO_LIST=(ASPECT) ;;
        aspectsin|sin)        TOPO_LIST=(ASPECT_SIN) ;;
        aspectcos|cos)        TOPO_LIST=(ASPECT_COS) ;;
        hillshade|hs)         TOPO_LIST=(HILLSHADE) ;;
        kombinasi|all|multimodal) TOPO_LIST=(DTM SLOPE ASPECT_COS ASPECT_SIN HILLSHADE) ;;
        *)
            echo "ERROR: --feature '$FEATURE' tidak dikenal." >&2
            echo "Pilih: dtm|slope|aspect|aspectsin|aspectcos|hillshade|kombinasi, atau pakai --topo." >&2
            exit 2 ;;
    esac
fi
if [[ ${#TOPO_LIST[@]} -eq 0 ]]; then
    echo "ERROR: tidak ada modality topografi." >&2
    exit 2
fi
TOPO_SLUG="$(printf '%s-' "${TOPO_LIST[@]}" | tr '[:upper:]' '[:lower:]' | sed 's/-$//')"

# -----------------------------------------------------------------------------
# Resolusi path dataset (default Google Drive, fallback lokal)
# -----------------------------------------------------------------------------
DATASET="$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')"

# Auto-deteksi split dari --data-root jika diberikan
if [[ -n "$DATA_ROOT" && -d "$DATA_ROOT" ]]; then
    for c in "$DATA_ROOT/3.TRAINING" "$DATA_ROOT/train" "$DATA_ROOT/TRAIN"; do
        if [[ -d "$c" ]]; then TRAIN_ROOT="$c"; break; fi
    done
    for c in "$DATA_ROOT/4.VALIDATION" "$DATA_ROOT/val" "$DATA_ROOT/VAL"; do
        if [[ -d "$c" ]]; then VAL_ROOT="$c"; break; fi
    done
    for c in "$DATA_ROOT/5.TESTING" "$DATA_ROOT/test" "$DATA_ROOT/TEST"; do
        if [[ -d "$c" ]]; then TEST_ROOT="$c"; break; fi
    done
fi

# Fallback ke dataset lokal repositori jika path Drive tidak ada
if [[ ! -d "$TRAIN_ROOT" && -d "$REPO_DIR/datasets/landslide/train" ]]; then
    TRAIN_ROOT="$REPO_DIR/datasets/landslide/train"
fi
if [[ ! -d "$VAL_ROOT" && -d "$REPO_DIR/datasets/landslide/val" ]]; then
    VAL_ROOT="$REPO_DIR/datasets/landslide/val"
fi
if [[ ! -d "$TEST_ROOT" && -d "$REPO_DIR/datasets/landslide/test" ]]; then
    TEST_ROOT="$REPO_DIR/datasets/landslide/test"
fi

# Resolusi blacklist
if [[ -z "$BLACKLIST" || "$BLACKLIST" == "auto" ]]; then
    BLACKLIST="none"
    for c in "/content/drive/MyDrive/project/black_list.txt" \
             "$REPO_DIR/datasets/landslide/black_list.txt" \
             "datasets/landslide/black_list.txt"; do
        if [[ -f "$c" ]]; then BLACKLIST="$c"; break; fi
    done
fi

if [[ "$DATASET" == "landslide" ]]; then
    for d in "$TRAIN_ROOT" "$VAL_ROOT"; do
        if [[ ! -d "$d" ]]; then
            echo "ERROR: folder dataset tidak ditemukan: $d" >&2
            echo "Set --train-root/--val-root atau --data-root." >&2
            exit 1
        fi
    done
elif [[ "$DATASET" == "pascal" ]]; then
    missing=0
    for pair in "train-img-dir:$P_TRAIN_IMG_DIR" "train-mask-dir:$P_TRAIN_MASK_DIR" \
                "train-split:$P_TRAIN_SPLIT" "val-img-dir:$P_VAL_IMG_DIR" \
                "val-mask-dir:$P_VAL_MASK_DIR" "val-split:$P_VAL_SPLIT"; do
        name="${pair%%:*}"; val="${pair#*:}"
        if [[ -z "$val" ]]; then echo "ERROR: --$name wajib untuk --dataset pascal." >&2; missing=1; fi
    done
    if [[ $missing -ne 0 ]]; then exit 1; fi
else
    echo "ERROR: --dataset harus 'landslide' atau 'pascal' (dapat: $DATASET)." >&2
    exit 2
fi

# Output folder (fallback lokal jika Drive tidak tersedia)
if [[ ! -d "$(dirname "$OUTPUT_DIR")" && ! -d "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$REPO_DIR/runs"
fi
mkdir -p "$OUTPUT_DIR"

# -----------------------------------------------------------------------------
# Checkpoint SAM2 (default Drive, fallback lokal, auto-download)
# -----------------------------------------------------------------------------
ensure_checkpoint() {
    local target="$1" url="$2" desc="$3"
    if [[ -f "$target" ]]; then
        echo "[$desc] ditemukan: $target"
        return 0
    fi
    echo "[$desc] tidak ada. Mengunduh: $url"
    mkdir -p "$(dirname "$target")"
    if command -v wget >/dev/null 2>&1; then
        wget -c "$url" -O "$target"
    elif command -v curl >/dev/null 2>&1; then
        curl -L "$url" -o "$target"
    else
        echo "ERROR: wget/curl tidak tersedia untuk unduh $desc." >&2
        return 1
    fi
}

if [[ ! -f "$HIERA_PATH" && -f "$REPO_DIR/checkpoints/sam2_hiera_large.pt" ]]; then
    HIERA_PATH="$REPO_DIR/checkpoints/sam2_hiera_large.pt"
fi
if [[ ! -f "$HIERA_PATH" && "$HIERA_PATH" == /content/* && ! -d "$(dirname "$HIERA_PATH")" ]]; then
    HIERA_PATH="$REPO_DIR/checkpoints/sam2_hiera_large.pt"
fi
ensure_checkpoint "$HIERA_PATH" \
    "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt" \
    "SAM2 Hiera Large"

# -----------------------------------------------------------------------------
# Nama run & folder output
# -----------------------------------------------------------------------------
if [[ -z "$RUN_NAME" ]]; then
    RUN_NAME="s2f_${DATASET}_${TOPO_SLUG}_shot${SHOT}"
fi
if [[ -z "$SAVE_PATH" ]]; then
    SAVE_PATH="$OUTPUT_DIR/$RUN_NAME"
fi
mkdir -p "$SAVE_PATH"

if [[ "$DATASET" == "pascal" ]]; then
    TAG="fold$FOLD"
else
    TAG="landslide"
fi

# -----------------------------------------------------------------------------
# Bangun command training
# -----------------------------------------------------------------------------
CMD=(
    "$PYTHON_BIN" train_fss.py
    --dataset "$DATASET"
    --hiera_path "$HIERA_PATH"
    --save_path "$SAVE_PATH"
    --shot "$SHOT"
    --size "$SIZE"
    --reduce_dim "$REDUCE_DIM"
    --epochs "$EPOCHS"
    --episodes_per_epoch "$EPISODES_PER_EPOCH"
    --val_episodes "$VAL_EPISODES"
    --val_interval "$VAL_INTERVAL"
    --batch_size "$BATCH_SIZE"
    --lr "$LR"
    --weight_decay "$WEIGHT_DECAY"
    --num_workers "$NUM_WORKERS"
    --seed "$SEED"
    --amp_dtype "$AMP_DTYPE"
)
if [[ -n "$CACHE_DIR" ]]; then
    CMD+=(--cache_dir "$CACHE_DIR")
fi

if [[ "$DATASET" == "landslide" ]]; then
    CMD+=(--data_dir "${DATA_ROOT:-$REPO_DIR/datasets/landslide}")
    CMD+=(--train_dir "$TRAIN_ROOT" --val_dir "$VAL_ROOT")
    CMD+=(--topo "${TOPO_LIST[@]}")
    CMD+=(--blacklist "$BLACKLIST")
else
    CMD+=(--fold "$FOLD")
    CMD+=(--train_img_dir "$P_TRAIN_IMG_DIR" --train_mask_dir "$P_TRAIN_MASK_DIR" --train_split "$P_TRAIN_SPLIT")
    CMD+=(--val_img_dir "$P_VAL_IMG_DIR" --val_mask_dir "$P_VAL_MASK_DIR" --val_split "$P_VAL_SPLIT")
fi

echo "============================================================================="
echo "S2F Few-Shot Training"
echo "  Dataset        : $DATASET"
if [[ "$DATASET" == "landslide" ]]; then
    echo "  Train root     : $TRAIN_ROOT"
    echo "  Val root       : $VAL_ROOT"
    echo "  Test root      : $TEST_ROOT"
    echo "  Topo modality  : ${TOPO_LIST[*]}"
    echo "  Blacklist      : $BLACKLIST"
else
    echo "  Fold           : $FOLD"
fi
echo "  Shot / Size    : $SHOT / $SIZE"
echo "  Epochs / Batch : $EPOCHS / $BATCH_SIZE"
echo "  LR / WD        : $LR / $WEIGHT_DECAY"
echo "  SAM2 checkpoint: $HIERA_PATH"
echo "  Save path      : $SAVE_PATH"
echo "============================================================================="
echo ">> ${CMD[*]}"
echo "-----------------------------------------------------------------------------"

cd "$REPO_DIR"
"${CMD[@]}"

# -----------------------------------------------------------------------------
# Evaluasi test opsional
# -----------------------------------------------------------------------------
if [[ "$RUN_TEST" == "true" ]]; then
    if [[ -z "$TEST_CHECKPOINT" ]]; then
        TEST_CHECKPOINT="$SAVE_PATH/S2F-$TAG-best.pth"
    fi
    if [[ ! -f "$TEST_CHECKPOINT" ]]; then
        echo "PERINGATAN: checkpoint test tidak ditemukan: $TEST_CHECKPOINT" >&2
        exit 0
    fi
    if [[ "$DATASET" == "landslide" && ! -d "$TEST_ROOT" ]]; then
        echo "PERINGATAN: test root tidak ditemukan: $TEST_ROOT, evaluasi test dilewati." >&2
        exit 0
    fi

    TEST_CMD=(
        "$PYTHON_BIN" test_fss.py
        --dataset "$DATASET"
        --checkpoint "$TEST_CHECKPOINT"
        --shot "$SHOT"
        --size "$SIZE"
        --reduce_dim "$REDUCE_DIM"
        --episodes "$VAL_EPISODES"
        --num_workers "$NUM_WORKERS"
        --amp_dtype "$AMP_DTYPE"
        --seed "$SEED"
    )

    if [[ "$DATASET" == "landslide" ]]; then
        TEST_CMD+=(--data_dir "${DATA_ROOT:-$REPO_DIR/datasets/landslide}")
        TEST_CMD+=(--test_dir "$TEST_ROOT")
        TEST_CMD+=(--topo "${TOPO_LIST[@]}" --blacklist "$BLACKLIST")
    else
        TEST_CMD+=(--fold "$FOLD")
        TEST_CMD+=(--img_dir "$P_TEST_IMG_DIR" --mask_dir "$P_TEST_MASK_DIR" --split "$P_TEST_SPLIT")
    fi

    echo ""
    echo ">> Evaluasi test: ${TEST_CMD[*]}"
    "${TEST_CMD[@]}"
fi
