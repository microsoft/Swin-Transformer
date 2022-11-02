# Fuzzy Fig

This experiment is the default swin-v2-base applied to iNat 21 192x192 with 1/4 the default learning rate (1.25e-4).
We train for 90 epochs at 192x192, then tune for 30 epochs at 256x256.

```yaml
configs: 
- configs/hierarchical-vision-project/fuzzy-fig-192.yaml
codename: fuzzy-fig
```

## Log

Training is running on 8x V100 as fuzzy-fig-192.
