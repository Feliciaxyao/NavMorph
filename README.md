# NavMorph
NavMorph: A Self-Evolving World Model for Vision-and-Language Navigation in Continuous Environments

## Introduction
![image](img/EWM.png)


## Usage

### Prerequisites

1. Follow the [Habitat Installation Guide](https://github.com/facebookresearch/habitat-lab#installation) and [VLN-CE](https://github.com/jacobkrantz/VLN-CE) to install [`habitat-lab`](https://github.com/facebookresearch/habitat-lab) and [`habitat-sim`](https://github.com/facebookresearch/habitat-sim). We use version `v0.2.1` in our experiments.
   
2. Install `torch_kdtree` and `tinycudann`: follow instructions [here](https://github.com/MrZihan/Sim2Real-VLN-3DFF). 

3. Install requirements:
```setup
conda create --name morph python=3.7.11
conda activate morph
```
* Required packages are listed in `environment.yaml`. You can install by running:

```
conda env create -f environment.yaml
```
4. Please download the preprocessed data and checkpoints from [GoogleDrive](https://drive.google.com/drive/folders/1w2-rFj1IshOEG5WDhlyl-pCUXeBYt_ls?usp=drive_link)

### Online Evaluation

Use pseudo interative demonstrator to equip the model with our NavMorph:
```
cd Nav_Morph
bash run_r2r/main.bash eval 
```




## Notice:
Our codes are uploaded only for peer review, please do not distribute them. The code is used to reproduce our experimental results on R2R-CE dataset.
The complete code will be released if the paper is accepted.

