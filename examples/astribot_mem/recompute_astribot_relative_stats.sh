#!/usr/bin/env bash
# Recompute MEAN/STD in relative-action space (grippers kept absolute).
# Run this after convert_astribot_data.sh and before train_pi05_mem.sh.

lerobot-edit-dataset \
    --repo_id=astribot_mem \
    --root=/home/phuclh12/data/Kronze157/astri_making_coffee_186_v21 \
    --new_repo_id=astribot_mem \
    --new_root=/home/phuclh12/data/Kronze157/astri_making_coffee_186_v21 \
    --operation.type=recompute_stats \
    --operation.overwrite=true \
    --operation.relative_action=true \
    --operation.relative_exclude_joints='["gripper"]' \
    --operation.chunk_size=50 \
    --operation.num_workers=4 \
    --push_to_hub=false
