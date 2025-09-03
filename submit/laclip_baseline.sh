#!/bin/bash -l

#SBATCH -o /ptmp/dduka/work/logs/laclip/%A_%a_%x_%j_%N.out
#SBATCH -e /ptmp/dduka/work/logs/laclip/%A_%a_%x_%j_%N.err

#SBATCH --job-name=laclip_baseline

#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --constraint="gpu"

#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=500000

#SBATCH --time=23:59:59
#SBATCH --wait-all-nodes=1

module purge
module load anaconda/3/2023.03

conda activate slip

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Set up distributed training environment variables
export MASTER_PORT=$((12000 + $RANDOM % 20000))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# Debug: Print GPU and node information
echo "Job running on nodes: $SLURM_JOB_NODELIST"
echo "Total nodes: $SLURM_NNODES" 
echo "GPUs per node: 4"
echo "Total GPUs: $((SLURM_NNODES * 4))"

cd /u/dduka/work/projects/LaCLIP/

TRAIN_NUM_SAMPLES=10968539

srun --cpu_bind=v --accel-bind=gn torchrun \
    --nproc_per_node=4 \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend=c10d \
    -m train \
    --train-data '/ptmp/dduka/databases/cc12m/data/cc12m-train-{0000..2175}.tar' \
    --train-num-samples $TRAIN_NUM_SAMPLES \
    --model CLIP_VITB16 \
    --lr 1e-3 \
    --wd 0.5 \
    --batch-size 128 \
    --epochs 35 \
    --warmup-epochs 1 \
    --aug-text \
    --output-dir '/ptmp/dduka/work/training_metadata/slip/LaCLIP_Baseline_ViTB16'