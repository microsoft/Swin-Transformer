# Groovy Grape

This experiment trains swin-v2-base from scratch on iNat21 using the multitask objective using 1/4 of the default learning rate, which works out to 1.25e-4.
It also does 90 epochs at 192, then 30 epochs at 256.

```yaml
configs:
- configs/swinv2/swinv2_base_patch4_window12_192_inat21_hierarchical_lr1.25.yaml
- configs/swinv2/swinv2_base_patch4_window12to16_192to256_inat21_hierarchical_lr1.25_ft.yaml
codename: groovy-grape
```

## Log

This model trained the first 90 on 8x V100, and did 8/30 epochs at 256 on 8x V100. 
I am storing the 8th checkpoint on S3 as groovy-grape-256-epoch8.pth.
Now it is stored at /local/scratch/stevens.994/hierarchical-vision/groovy-grape-256/v0
It was originally haunted-broomstick on wandb, but is now groovy-grape-256.
