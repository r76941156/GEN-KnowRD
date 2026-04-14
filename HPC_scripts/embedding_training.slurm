#!/bin/bash
#SBATCH --job-name=embedding_training
#SBATCH --mail-user=your@email.account
#SBATCH --mail-type=ALL
#SBATCH --account=your_HPC_account
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_h100_nvl:2
#SBATCH --time=24:00:00
#SBATCH --partition=batch_gpu
#SBATCH --output=training_result.txt

# === Environment ===
setup_accre_software_stack
module load python/3.12.4
module load cuda/12.6
module load scipy-stack/2025a
source qwen/bin/activate

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHELASTIC_ERROR_FILE=traceback.log
export SWIFT_DEBUG=true

nproc_per_node=2
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model Qwen/Qwen3-Embedding-8B \
    --task_type embedding \
    --model_type qwen3_emb \
    --train_type full \
    --dataset ./disease_sections_claude.jsonl \
    --output_dir /output_data/qwen_emb8b_claude_kb \
    --split_dataset_ratio 0.1 \
    --eval_strategy steps \
    --eval_steps 10 \
    --num_train_epochs 1 \
    --save_steps 50 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-6 \
    --loss_type infonce \
    --dataloader_drop_last true \
    --deepspeed zero3 \
    --bf16 true \
