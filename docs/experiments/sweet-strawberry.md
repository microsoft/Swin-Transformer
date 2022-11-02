# Sweet Strawberry

This experiment trains swin-v2-base from scratch on iNat21 using the multitask objective using 1/2 of the default learning rate, which works out to 2.5e-4.
It also does 90 epochs at 192.

```yaml
configs:
- configs/hierarchical-vision-project/sweet-strawberry-192.yaml
codename: sweet-strawberry
```

## Log

This model trained the first 90 on 8x V100, and did 15/30 epochs at 256 on 8x V100. 
I am storing the 15th checkpoint on S3 as sweet-strawberry-256-epoch15.pth.
Now it is stored at /local/scratch/stevens.994/hierarchical-vision/sweet-strawberry-256/v0
It was originally unearthly-moon on wandb, but is now sweet-strawberry-256.
It is running on 4x A6000.

I want to check that I didn't mix up sweet-strawberry-192 and groovy-grape-192, so I am going to check their validation top 1 accuracy on iNat21 192x192 using their final checkpoints.
