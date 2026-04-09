#!/bin/bash
#SBATCH --job-name=morpheus-rl
#SBATCH --account=def-sponsor00
#SBATCH --partition=nodegpupool
#SBATCH --gres=gpu:1g.5gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kains.praveen.kavuluri@umontreal.ca

# ─── Environment ──────────────────────────────────────────────────────────────
module load python/3.11 scipy-stack

source /home/kainsk1/morpheus/.venv/bin/activate

export SLURM_CACHE=$SLURM_TMPDIR/cache
mkdir -p $SLURM_CACHE

# HuggingFace / Transformers
export HF_HOME=$SLURM_CACHE/hf
export TRANSFORMERS_CACHE=$SLURM_CACHE/hf
export HF_DATASETS_CACHE=$SLURM_CACHE/hf/datasets

export PIP_CACHE_DIR=$SLURM_CACHE/pip

export WANDB_DIR=$SLURM_CACHE/wandb
export WANDB_CACHE_DIR=$SLURM_CACHE/wandb/cache
export WANDB_DATA_DIR=$SLURM_CACHE/wandb/data

# Misc Python caches
export MPLCONFIGDIR=$SLURM_CACHE/matplotlib
export NUMBA_CACHE_DIR=$SLURM_CACHE/numba
export XDG_CACHE_HOME=$SLURM_CACHE/xdg

HF_MODEL_SRC=${SCRATCH:-$HOME/.cache/huggingface/hub}/models--allenai--scibert_scivocab_uncased
if [ -d "$HF_MODEL_SRC" ]; then
    mkdir -p $HF_HOME/hub
    cp -r "$HF_MODEL_SRC" $HF_HOME/hub/
fi

pip install -q nltk python-Levenshtein fcd_torch --no-index 2>/dev/null || \
pip install -q nltk python-Levenshtein fcd_torch 2>/dev/null || true
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)" 2>/dev/null

if [ -n "$WANDB_API_KEY" ]; then
    wandb login "$WANDB_API_KEY"
fi

# ─── Project directory ────────────────────────────────────────────────────────
cd /home/kainsk1/morpheus              # ← adjust to your project path
mkdir -p logs

# ─── Algorithm ────────────────────────────────────────────────────────────────
ALGORITHM=${1:-reinforce}

# ─── Run ──────────────────────────────────────────────────────────────────────
python rl/train_prompt_adherence.py \
    --algorithm  $ALGORITHM \
    --wandb_artifact marl-project/morpheus-diffusion-finetune-contrastive/finetuned-model-valloss0.5950:v0 \
    --text_encoder_artifact marl-project/morpheus-contrastive/contrastive-text-selfies-8wf0yquy:v0 \
    --data_path  train.csv \
    --test_data_path test.csv \
    --num_steps  500 \
    --batch_size 16 \
    --cfg_scale  3.0 \
    --gen_steps  15 \
    --eval_gen_steps 32 \
    --lr         1e-5 \
    --beta       0.1 \
    --temperature 0.9 \
    --w_tanimoto 1.0 \
    --w_bleu2    0.5 \
    --w_bleu4    0.5 \
    --eval_every 50 \
    --save_every 100 \
    --num_eval   100 \
    --run_name   "${ALGORITHM}_v1" \
    --output_dir outputs/rl_prompt \
    --group_size 4 \
    --ppo_epochs 4 \
    --clip_eps   0.2 \
    --wandb_project morpheus-rl \
    --chebi20_eval
