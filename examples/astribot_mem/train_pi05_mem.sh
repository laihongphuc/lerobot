#!/usr/bin/env bash
# LoRA finetune of Pi05 + MEM short-horizon visual memory on the local
# Astribot making-coffee dataset. Requires `lerobot[peft]`.
# Recompute relative-action stats first:
#   bash examples/astribot_mem/recompute_astribot_relative_stats.sh

lerobot-train \
    --dataset.repo_id=astribot_mem \
    --dataset.root=/home/phuclh12/data/Kronze157/astri_making_coffee_186_v30 \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.use_visual_memory=true \
    --policy.memory_frames=6 \
    --policy.memory_stride=30 \
    --policy.memory_temporal_attention_every=4 \
    --policy.use_relative_actions=true \
    --policy.relative_exclude_joints='["gripper"]' \
    --policy.normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}' \
    --policy.n_action_steps=50 \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.optimizer_lr=2.5e-4 \
    --policy.scheduler_decay_lr=2.5e-5 \
    --peft.method_type=LORA \
    --peft.r=64 \
    --peft.lora_alpha=64 \
    --output_dir=./outputs/pi05_mem_astribot_lora \
    --job_name=pi05_mem_astribot_lora \
    --batch_size=32 \
    --num_workers=8 \
    --steps=50000 \
    --save_freq=1000 \
    --seed=1000
