### Depth-aware CNN for RGB-D Segmentation*

[<a href="https://arxiv.org/pdf/1706.02413.pdf">Arxiv</a>]

### Installation
Install <a href="http://pytorch.org/">Pytorch</a>, <a href="https://github.com/Knio/dominate">dominate</a>, <a href="https://github.com/lanpa/tensorboard-pytorch">TensorBoardX</a>.

The depth-aware convolution and depth-aware average pooling operations are under folder `models/ops`, to build them, simply type `sh make.sh`.

### Training

```bash
#!./scripts/train.sh
python train.py \
--name nyuv2_VGGdeeplab_depthconv \
--dataset_mode nyuv2 \
--flip --scale --crop --colorjitter \
--depthconv \
--list ./lists/train.lst \
--vallist ./lists/val.lst
```

### Testing 

```bash
#!./scripts/test.sh
python train.py \
--name nyuv2_VGGdeeplab_depthconv \
--dataset_mode nyuv2 \
--flip --scale --crop --colorjitter \
--depthconv \
--list ./lists/train.lst \
--vallist ./lists/val.lst
```

### Citation

        @article{qi2017pointnetplusplus,
          title={PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
          author={Qi, Charles R and Yi, Li and Su, Hao and Guibas, Leonidas J},
          journal={arXiv preprint arXiv:1706.02413},
          year={2017}
        }

### Acknowledgemets

The visulization code are borrowed from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
