# Outrageous Orange

This experiment is the swin-v2-base with hierarchical multitask objective applied to iNat 21 192x192.
We train for 90 epochs at 192x192, then tune for 30 epochs at 256x256.

```yaml
configs: 
- configs/swinv2/swinv2_base_patch4_window12_192_inat21_hierarchical_lr5.yaml
codename: outrageous-orange
```

## Log

I initialized training on strawberry0 on 4x A6000 servers.

I decided to use the A6000 servers for 256x256 tuning, so I am moving the latest checkpoint to S3, then cloning it back to an 8x V100 server to finish training.
I am storing the 36th checkpoint on S3 as `outrageous-orange-192-epoch36.pth`.
It is now running as `outrageous-orange-192` on 8x V100.
