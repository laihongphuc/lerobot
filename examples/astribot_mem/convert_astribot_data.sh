#!/usr/bin/env bash

python src/lerobot/scripts/convert_dataset_v21_to_v30.py \
    --repo-id=astribot_mem \
    --root=/home/phuclh12/data/Kronze157/astri_making_coffee_186_v21 \
    --push-to-hub=false
