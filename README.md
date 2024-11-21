# NavMorph
NavMorph: A Self-Evolving World Model for Vision-and-Language Navigation in Continuous Environments

## Introduction
![image](img/EWM.pdf)


## Usage

### Prerequisites

1. Install Matterport3D simulators: follow instructions [here](https://github.com/peteanderson80/Matterport3DSimulator). We use the latest version the same as DUET.
```
export PYTHONPATH=Matterport3DSimulator/build:$PYTHONPATH
```

2. Install requirements:
```setup
conda create --name fsvln python=3.8.5
conda activate fsvln
```
* Required packages are listed in `requirements.txt`. You can install by running:

```
pip install -r requirements.txt
```

3. Please download data from [Dropbox](https://www.dropbox.com/sh/u3lhng7t2gq36td/AABAIdFnJxhhCg2ItpAhMtUBa?dl=0), including processed annotations, features and pretrained models of REVERIE datasets and R2R datasets. 
Before running the code, please put the data in `datasets' directory.

4. Please download pretrained LXMERT model by running:
```
mkdir -p datasets/pretrained 
wget https://nlp.cs.unc.edu/data/model_LXRT.pth -P datasets/pretrained
```


### Pretraining (Base Model)

Combine behavior cloning and auxiliary proxy tasks in pretraining:
```pretrain
cd pretrain_src
bash run_reverie.sh 
```



### Fine-tuning (Base Model)

Use pseudo interative demonstrator to fine-tune the model:
```finetune
cd map_nav_src
bash scripts/run_reverie.sh 
```

### Test-time Adaptation & Evaluation

Use pseudo interative demonstrator to equip the model with our FSTTA:
```TTA during test time
cd map_nav_src
bash scrips/run_reverie_tta.sh
```


## Acknowledgements
Our implementations are partially based on [VLN-DUET](https://github.com/cshizhe/VLN-DUET), [HM3DAutoVLN](https://github.com/cshizhe/HM3DAutoVLN) and [VLN-BEVBert](https://github.com/MarSaKi/VLN-BEVBert.git). Thanks to the authors for sharing their code.


## Related Work
* [Reverie: Remote embodied visual referring expression in real indoor environments](https://openaccess.thecvf.com/content_CVPR_2020/papers/Qi_REVERIE_Remote_Embodied_Visual_Referring_Expression_in_Real_Indoor_Environments_CVPR_2020_paper.pdf)
* [Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments](https://arxiv.org/pdf/2004.02857)



