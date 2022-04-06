# SurroundDepth
### [Project Page](https://weiyithu.github.io/NerfingMVS) | [Paper](https://arxiv.org/abs/2109.01129) | [Data](https://cloud.tsinghua.edu.cn/d/8c4541693833471e8b13/)
<br/>

> SurroundDepth: Entangling Surrounding Views for Self-Supervised Multi-Camera Depth Estimation       
> [Yi Wei*](https://weiyithu.github.io/), [Linqing Zhao*](https://github.com/lqzhao), [Wenzhao Zheng](https://scholar.google.com/citations?user=LdK9scgAAAAJ&hl=en), [Zheng Zhu](http://www.zhengzhu.net/), [Yongming Rao](https://raoyongming.github.io/), Guan Huang, [Jiwen Lu](http://ivg.au.tsinghua.edu.cn/Jiwen_Lu/), [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1)  

<p align='center'>
<img src="https://github.com/weiyithu/weiyithu.github.io/blob/main/images/surrounddepth.gif" width='80%'/>
</p>


## Introduction
Depth estimation from images serves as the fundamental step of 3D perception for autonomous driving and is an economical alternative to expensive depth sensors like LiDAR. The temporal photometric consistency enables self-supervised depth estimation without labels, further facilitating its application. However, most existing methods predict the depth solely based on each monocular image and ignore the correlations among multiple surrounding cameras, which are typically available for modern self-driving vehicles. In this paper, we propose a SurroundDepth method to incorporate the information from multiple surrounding views to predict depth maps across cameras.

## Model Zoo

| type     | dataset | Abs Rel | Sq Rel | delta < 1.25 | download |  
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| scale-ambiguous | DDAD | 0.200  | 3.392 | 0.740 | [model]() |
| scale-aware | DDAD | 0.208  | 3.371 | 0.693 | [model]() |
| scale-ambiguous | nuScenes | 0.245  | 3.067 | 0.719 | [model]() |
| scale-aware | nuScenes | 0.280  | 4.401 | 0.661 | [model]() |

## Install
* python 3.8, pytorch 1.8.1, CUDA 11.4, RTX 3090
```bash
git clone https://github.com/weiyithu/SurroundDepth.git
conda create -n surrounddepth python=3.8
conda activate surrounddepth
pip install -r requirements.txt
```
Since we use [dgp codebase](https://github.com/TRI-ML/dgp) to generate groundtruth depth, you should also install it. 

## Data Preparation
Datasets are assumed to be downloaded under `data/<dataset-name>`.

### DDAD
* Please download the official [DDAD dataset](https://tri-ml-public.s3.amazonaws.com/github/DDAD/datasets/DDAD.tar) and place them under `data/ddad/raw_data`. You may refer to official [DDAD repository](https://github.com/TRI-ML/DDAD) for more info and instructions.
* Please download [metadata]() of DDAD and place these pkl files in `datasets/ddad`.
* We provide annotated self-occlusion masks for each sequences. Please download [masks]() and place them in `data/ddad/mask`.
* Export depth maps for evaluation 
```bash
cd tools
python export_gt_depth_ddad.py val
```
* Generate scale-aware SfM pseudo labels for pretraining (it may take several hours). Note that since we use SIFT descriptor in cv2, you need to change cv2 version and we suggest start a new conda environment.
```bash
conda create -n sift python=3.6
conda activate sift
pip install opencv-python==3.4.2.16 opencv-contrib-python==3.4.2.16
python sift_ddad.py
python match_ddad.py
```
* The final data structure should be:
```
SurroundDepth
├── data
│   ├── ddad
│   │   │── raw_data
│   │   │   │── 000000
|   |   |   |── ...
|   |   |── depth
│   │   │   │── 000000
|   |   |   |── ...
|   |   |── match
│   │   │   │── 000000
|   |   |   |── ...
|   |   |── mask
│   │   │   │── 000000
|   |   |   |── ...
```

### nuScenes
* Please download the official [nuScenes dataset](https://www.nuscenes.org/download) to `data/nuscenes/raw_data`
* Export depth maps for evaluation 
```bash
cd tools
python export_gt_depth_nusc.py val
```
* Generate scale-aware SfM pseudo labels for pretraining (it may take several hours).
```bash
conda activate sift
python sift_nusc.py
python match_nusc.py
```
* The final data structure should be:
```
SurroundDepth
├── data
│   ├── nuscenes
│   │   │── raw_data
│   │   │   │── samples
|   |   |   |── sweeps
|   |   |   |── maps
|   |   |   |── v1.0-trainval
|   |   |── depth
│   │   │   │── samples
|   |   |── match
│   │   │   │── samples
```

## Training
Take DDAD dataset as an example. 
Train scale-ambiguous model.
```bash
python -m torch.distributed.launch --nproc_per_node 8 --num_workers=8 run.py  --model_name ddad  --config configs/ddad.txt 
```
Train scale-aware model. First we should conduct SfM pretraining.
```bash
python -m torch.distributed.launch --nproc_per_node 8  run.py  --model_name ddad_scale_pretrain  --config configs/ddad_scale_pretrain.txt 
```
Then we select the best pretrained model.
```bash
python -m torch.distributed.launch --nproc_per_node 8  run.py  --model_name ddad_scale  --config configs/ddad_scale.txt  --load_weights_folder=${best pretrained}
```
We observe that the training on nuScenes dataset is unstable and easy to overfit. Also, the results with 4 GPUs are much better than 8 GPUs. Thus, we set fewer epochs and use 4 GPUs for nuScenes experiments. We also provide SfM pretrained model on [DDAD]() and [nuScenes]().  

## Evaluation
```bash
python -m torch.distributed.launch --nproc_per_node ${NUM_GPU}  run.py  --model_name test  --config configs/${TYPE}.txt --models_to_load depth encoder   --load_weights_folder=${PATH}  --eval_only 
```

## Acknowledgement

Our code is based on [Monodepth2](https://github.com/nianticlabs/monodepth2).

## Citation

If you find this project useful in your research, please consider cite:
```
@article{wei2022surround,
    title={SurroundDepth: Entangling Surrounding Views for Self-Supervised Multi-Camera Depth Estimation},
    author={Wei, Yi and Zhao, Linqing and Zheng, Wenzhao and Zhu, Zheng and Rao, Yongming and Huang ,Guan and Lu, Jiwen and Zhou, Jie},
    journal={arXiv preprint arXiv:2022.6980},
    year={2022}
}
```


