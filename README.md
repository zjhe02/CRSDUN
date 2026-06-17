## [CVPR 2026] Joint Spectral Image Reconstruction and Semantic Segmentation with Cooperative Unfolding

This repo is the implementation of the paper "[Joint Spectral Image Reconstruction and Semantic Segmentation with Cooperative Unfolding](https://openaccess.thecvf.com/content/CVPR2026/html/He_Joint_Spectral_Image_Reconstruction_and_Semantic_Segmentation_with_Cooperative_Unfolding_CVPR_2026_paper.html)".

## Abstract

Coded Aperture Snapshot Spectral Imaging (CASSI) is an emerging hyperspectral image (HSI) acquisition technique for downstream semantic segmentation. Due to the ill-posedness nature of CASSI systems, typical solutions are compelled to conduct a two-stage reconstruction then-segmentation pipeline, namely viewing them as two separate tasks. However, we observe that such two tasks are interrelated and mutually reinforcing for representation learning, and thus separating them limits the overall accuracy and efficiency. To this end, we propose the first **C**ooperative **R**econstruction-**S**egmentation **D**eep **U**nfolding **N**etwork (**CRSDUN**) to solve the reconstruction and segmentation tasks in parallel. To make the two mutually reinforcing, we introduce the Cross-Aggregated Super-Token Attention (CASTA) mechanism to enhance the representation interactions between HSI reconstruction and semantic segmentation. Extensive experiments on both synthetic and real-world HSI reconstruction-segmentation datasets demonstrate that our method achieves state-of-the-art in both spectral reconstruction and semantic segmentation.

## Dataset

Download  ([FVgNET dataset](https://pan.baidu.com/s/1Z69PEVDEx-rxd4-YVByzmw?pwd=jjkf), code: `fo0q` ), and then put them into the corresponding folders of `fvgnet/` as follows:

```
|--fvgnet
	|--labels
		|--2021-11-03_006.png
		|--2021-11-03_007.png
		： 
		|--2021-11-10_050.png
	|--visible_28
		|--2021-11-03_006.npy
		|--2021-11-03_007.npy
		： 
		|--2021-11-10_050.npy
	|--test_data.csv
	|--train_data.csv
```

## Pretrained model and Results

The checkpoint and results are publicly accessible at → [Pretrained model and Results](https://pan.baidu.com/s/1tz1O-1yeSwhHAL1Tu0sD7A?pwd=m97x), code: `m97x`


## Training

```
bash train.sh
```

The training log and trained model will be available in `./exp/` .

## Testing

Place the pretrained model to `./checkpoint/`

Run the following command to test the model on FVgNET

```
bash test.sh
```

The reconstructed HSIs and segmentation results will be output into `./exp/CRSDUN_test/`


## Acknowledgements

Our code and dataset are based on the following works, thanks to their generous open source:

- [MST](https://github.com/caiyuanhao1998/MST)
- [Hyplex](https://github.com/makamoa/hyplex)
- [STViT](https://github.com/hhb072/STViT)
- [SSR](https://github.com/ZhangJC-2k/SSR)
- [Swin-Transformer](https://github.com/microsoft/Swin-Transformer/tree/main)


## Citation

If this code helps you, please consider citing our work:

```
@InProceedings{crsdun_cvpr2026,
    author={He, Zijun and Wang, Ping and Wang, Xiaodong and Chen, Chang and Yuan, Xin},
    title={Joint Spectral Image Reconstruction and Semantic Segmentation with Cooperative Unfolding},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month={June},
    year={2026},
    pages={6910-6919}
}
```
