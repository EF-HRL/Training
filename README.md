# EF-HRL Training Code

This directory contains the training code for *Bridging Nav–Loco Mismatch: Explicit Feedback for Hierarchical Reinforcement Learning Based Quadruped Navigation*.

Our implementation builds on REASAN. We keep its pretrained navigation and locomotion policies frozen and use them to collect velocity-tracking data. We train a Smooth Neural Surrogate (SNS) on these data to predict tracking errors and use its gradient to adjust navigation commands online.

## Training workflow

Run both commands from the `training` directory.

### 1. Collect the dataset

```bash
python scripts/collect_loco_data.py --headless --load_run nav_1 \
  --num_envs 64 --num_samples 8192000 \
  --filename loco_error_dataset_nav_guided.h5
```

### 2. Train the SNS

```bash
python go2_lidar/go2_lidar/SNS/train_sns.py \
  --data loco_error_dataset_nav_guided.h5 \
  --device cuda --epochs 50 --batch-size 2048
```

## Acknowledgments

We thank the authors and contributors of [REASAN](https://github.com/ASIG-X/REASAN) for sharing their training framework, simulation environments, and pretrained policies.
