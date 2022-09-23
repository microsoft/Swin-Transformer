## Swin Transformer V2

| name | model size | pre-train dataset | pre-train iterations | validation loss | fine-tuned acc | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SwinV2-Small | 49M | ImageNet-1K 10% | 125k | - | - | [azure](azure) |


## Swin Transformer V1

**ImageNet-1K Pre-trained and Fine-tuned Models**

| name | pre-train epochs | pre-train resolution | fine-tune resolution | acc@1 | pre-trained model | fine-tuned model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Swin-Base | 100 | 192x192 | 192x192 | 82.8 | [google](https://drive.google.com/file/d/1Wcbr66JL26FF30Kip9fZa_0lXrDAKP-d/view?usp=sharing)/[config](configs/swin_base__100ep/simmim_pretrain__swin_base__img192_window6__100ep.yaml) | [google](https://drive.google.com/file/d/1RsgHfjB4B1ZYblXEQVT-FPX3WSvBrxcs/view?usp=sharing)/[config](configs/swin_base__100ep/simmim_finetune__swin_base__img192_window6__100ep.yaml) |
| Swin-Base | 100 | 192x192 | 224x224 | 83.5 | [google](https://drive.google.com/file/d/1Wcbr66JL26FF30Kip9fZa_0lXrDAKP-d/view?usp=sharing)/[config](configs/swin_base__100ep/simmim_pretrain__swin_base__img192_window6__100ep.yaml) | [google](https://drive.google.com/file/d/1mb43BkW56F5smwiX-g7QUUD7f1Rftq8u/view?usp=sharing)/[config](configs/swin_base__100ep/simmim_finetune__swin_base__img224_window7__100ep.yaml) |
| Swin-Base | 800 | 192x192 | 224x224 | 84.0 | [google](https://drive.google.com/file/d/15zENvGjHlM71uKQ3d2FbljWPubtrPtjl/view?usp=sharing)/[config](configs/swin_base__800ep/simmim_pretrain__swin_base__img192_window6__800ep.yaml) | [google](https://drive.google.com/file/d/1xEKyfMTsdh6TfnYhk5vbw0Yz7a-viZ0w/view?usp=sharing)/[config](configs/swin_base__800ep/simmim_finetune__swin_base__img224_window7__800ep.yaml) |
| Swin-Large | 800 | 192x192 | 224x224 | 85.4 | [google](https://drive.google.com/file/d/1qDxrTl2YUDB0505_4QrU5LU2R1kKmcBP/view?usp=sharing)/[config](configs/swin_large__800ep/simmim_pretrain__swin_large__img192_window12__800ep.yaml) | [google](https://drive.google.com/file/d/1mf0ZpXttEvFsH87Www4oQ-t8Kwr0x485/view?usp=sharing)/[config](configs/swin_large__800ep/simmim_finetune__swin_large__img224_window14__800ep.yaml) |
| SwinV2-Huge | 800 | 192x192 | 224x224 | 85.7 | / | / |
| SwinV2-Huge | 800 | 192x192 | 512x512 | 87.1 | / | / |