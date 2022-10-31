# Sweet Strawberry

This experiment trains swin-v2-base from scratch on iNat21 using the multitask objective using 1/2 of the default learning rate, which works out to 2.5e-4.
It also does 90 epochs at 192, then 30 epochs at 256.

```yaml
configs:
- configs/swinv2/swinv2_base_patch4_window12_192_inat21_hierarchical_lr2.5.yaml
- configs/swinv2/swinv2_base_patch4_window12to16_192to256_inat21_hierarchical_lr2.5_ft.yaml
codename: sweet-strawberry
```

## Log

This model trained the first 90 on 8x V100, and did 15/30 epochs at 256 on 8x V100. 
I am storing the 15th checkpoint on S3 as sweet-strawberry-256-epoch15.pth.
Now it is stored at /local/scratch/stevens.994/hierarchical-vision/sweet-strawberry-256/v0
It was originally unearthly-moon on wandb, but is now sweet-strawberry-256.
It is running on 4x A6000.
