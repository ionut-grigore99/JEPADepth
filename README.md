# JEPADepth


## Introduction
This is the implementation of the paper called "*JEPADepth: Some Title Here*".
<br />
<br />
![img.png](assets/JEPADepth_architecture.png)

If you find our work useful in your research please consider citing our paper:

```
@article{grigore2026hyenadepth,
  title={JEPADepth: Some Title Here},
  author={Grigore, Ionuț and Popa, Călin-Adrian},
  journal={arXiv preprint arXiv:...},
  year={2026}
}
```

## Theoretical prerequisites
Fundamentally, self-supervised depth estimation is a form of Structure from Motion (SfM), where the monocular camera is moving within a rigid 
environment to provide multiple views of that scene. 

The seminal work in this field is "*Digging Into Self-Supervised Monocular Depth Estimation*" (https://arxiv.org/pdf/1806.01260.pdf) upon which 
almost every self-supervised depth estimation method/paper/repo is based on (also our repo).

![img.png](assets/Theory_1_white.png)
![img_2.png](assets/Theory_2_white.png)

<br />
Self-supervised framework:

![img.png](assets/Self-supervised_framework.png)
## Instalation
Requirements: `Ubuntu 20.04`, `NVIDIA GPU`, `CUDA >= 11.7`

You'll probably find useful this documentation [Cuda-installation-guide-Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#conda-installation) and also
this:
```bash
conda install cuda -c nvidia
```

First create a conda environment called **jepadepth**:
```bash
conda create -n jepadepth  python=3.10  
```

Activate the new enviroment:
```bash
conda activate jepadepth
```

After that, run the following:
```bash
pip install -e .
```
or
```bash
pip install -e . && pip install -e ".[dev]"
```

Recommended to install the `[dev]` dependencies.

## KITTI training data
You can download the entire [raw KITTI dataset](https://www.cvlibs.net/datasets/kitti/raw_data.php) by running:
```bash
wget -i data/kitti/kitti_archives_to_download.txt -P data/kitti/kitti_data/
```
The **'-i'** option in wget stands for "input-file". <br />
This option specifies a file that contains a list of URLs to download.  <br />
In this case, the file is src/data/kitti/kitti_archives_to_download.txt.  <br />
This file should contain a list of URLs, each on a new line, pointing to the files that need to be downloaded. <br />

<br />

The **'-P'** option specifies the prefix (directory) where downloaded files will be saved. <br /> 
In this case, the files are being downloaded to the src/data/kitti/kitti_data/ directory. <br />

<br />
Then unzip with:

```bash
cd data/kitti/kitti_data
unzip "*.zip"
cd .. # 3 times
```
<br />

**Warning**: it weighs about 175GB, so make sure you have enough space to unzip too!
<br />

Their default settings expect that you have converted the png images to jpeg with this command, **which also deletes
the raw KITTI `.png` files**:
```bash
find data/kitti/kitti_data/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
```
**or** you can skip this conversion step and train from raw png files by adding the flag `--png` when training, at the expense of slower load times.

The above conversion command creates images which match their experiments (see [Monodepth2](https://github.com/nianticlabs/monodepth2)), where KITTI `.png` images were converted to `.jpg` on Ubuntu 16.04 
with default chroma subsampling 2x2,1x1,1x1. They found that Ubuntu 18.04 defaults to 2x2,2x2,2x2, which gives different results, hence 
the explicit parameter in the conversion command.

## Cityscapes evaluation data

TODO

## Make3D evaluation data

We use the Make3D test set for evaluating our method (zero-shot evaluation), which could be downloaded in [here](http://make3d.cs.cornell.edu/data.html).



## Inference

```bash
python -m src.inference.test  
```

## Evaluation

```bash
python -m src.evaluation.evaluate_depth
```

### Evaluation of the pretrained models

The structure of folder containing the pretrained weights should look like this:
<br /> TODO


#### Evaluation split: eigen


#### Evaluation split: eigen_benchmark


#### Cityscapes zero-shot evaluation

#### Make3D zero-shot evaluation


## Training
```bash
python -m src.train.trainer
```

## Local overfit

The goal was to see if the JEPADepth architecture could be overfit on a small batch of samples—typically one, two, or five images.
The ability to overfit is a litmus test for model capacity, and visual aids, such as the rendering of predicted depth maps, were instrumental in evaluating the success. 
If overfitting was achieved with satisfactory visual confirmation, the next logical step involved deploying the entire training pipeline, utilizing the full dataset.
```bash
python -m src.overfit.local_trainer
```

## Jupyter notebooks

You can also find some useful jupyter notebooks which have the purpose of illustrating the functionality of main parts of this project.

## Contact us

Contact us: ionut.grigore@cs.upt.ro