# Local Relation Networks V2 (LR-Net V2)

This branch is an improved implementation of ["Local Relation Networks for Image Recognition (LR-Net)"](https://arxiv.org/pdf/1904.11491.pdf). The original LR-Net utilizes sliding window based self-attention layer to replace the `3x3` convolution layers in a ResNet architecture. This improved implementation applies this layer into a stronger overall architecture based on Tranformers, dubbed as LR-Net V2. We provide cuda kernels for the local relation layers. Training scripts and pre-trained models will be provided in the future.

## Install 
```bash
cd ops/local_relation
python setup.py build_ext --inplace
```

## Citing Local Relation Networks

```
@inproceedings{hu2019local,
  title={Local relation networks for image recognition},
  author={Hu, Han and Zhang, Zheng and Xie, Zhenda and Lin, Stephen},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={3464--3473},
  year={2019}
}
```
```
@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
