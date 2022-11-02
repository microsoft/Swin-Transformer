# Outrageous Orange

This experiment is the swin-v2-base with hierarchical multitask objective applied to iNat 21 192x192.
We train for 90 epochs at 192x192.

```yaml
configs: 
- configs/hierarchical-vision-project/outrageous-orange-192.yaml
codename: outrageous-orange
```

## Log

I initialized training on strawberry0 on 4x A6000 servers.

I decided to use the A6000 servers for 256x256 tuning, so I am moving the latest checkpoint to S3, then cloning it back to an 8x V100 server to finish training.
I am storing the 36th checkpoint on S3 as `outrageous-orange-192-epoch36.pth`.
It is now running as `outrageous-orange-192` on 8x V100.

I am going to stop outrageous-orange-192 (and never run outrageous-orange-256) because it is underperforming the other hierarchical runs and I don't want to waste compute on an obviously bad run. I'll still upload the checkpoints to S3 to save them. 
The latest checkpoint is at `s3://imageomics-models/outrageous-orange-192-epoch64.pth`

This is a bad run (std/mean switching issue).
I am going to restart it on 4x A60000 because I expect it to underperform the other run.
