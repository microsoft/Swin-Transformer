# Funky Banana

This experiment is the default swin-v2-base applied to iNat 21 192x192.
We train for 90 epochs at 192x192, then tune for 30 epochs at 256x256.

```yaml
configs: 
- configs/swinv2/swinv2_base_patch4_window12_192_inat21.yaml
- configs/swinv2/swinv2_base_patch4_window12to16_192to256_inat21.yaml
codename: funky-banana
```

## Log

I initialized training on strawberry0 on 4x A6000 servers.

I decided to use the A6000 servers for 256x256 tuning, so I am moving the latest checkpoint to S3, then cloning it back to an 8x V100 server to finish training.
I am storing the 40th checkpoint on S3 as funky-banana-192-epoch40.pth.

