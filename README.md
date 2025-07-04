# NavMorph: A Self-Evolving World Model for Vision-and-Language Navigation in Continuous Environments

**Xuan Yao, Junyu Gao, and Changsheng Xu**

This repository is the official implementation of [NavMorph: A Self-Evolving World Model for Vision-and-Language Navigation in Continuous Environments](https://arxiv.org/abs/2506.23468).

> Vision-and-Language Navigation in Continuous Environments (VLN-CE) requires agents to execute sequential navigation actions in complex environments guided by natural language instructions. Current approaches often struggle with generalizing to novel environments and adapting to ongoing changes during navigation.
Inspired by human cognition, we present NavMorph, a self-evolving world model framework that enhances environmental understanding and decision-making in VLN-CE tasks. NavMorph employs compact latent representations to model environmental dynamics, equipping agents with foresight for adaptive planning and policy refinement. By integrating a novel Contextual Evolution Memory, NavMorph leverages scene-contextual information to support effective navigation while maintaining online adaptability. Extensive experiments demonstrate that our method achieves notable performance improvements on popular VLN-CE benchmarks.

![image](img/EWM.png)


## 🌍 Usage

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
   

### Dataset Preparation

1. **Scenes for Matterport3D**

   > Instructions copied from [VLN-CE](https://github.com/jacobkrantz/VLN-CE)

   Matterport3D (MP3D) scene reconstructions are used. The official Matterport3D download script (`download_mp.py`) can be accessed by following the instructions on their [project webpage](https://niessner.github.io/Matterport/). The scene data can then be downloaded:

   ```bash
   # requires running with python 2.7
   python download_mp.py --task habitat -o data/scene_datasets/mp3d/
   ```
   
   Extract such that it has the form `scene_datasets/mp3d/{scene}/{scene}.glb`. There should be 90 scenes. Place the `scene_datasets` folder in `data/`.

2. **Data and Trained Models**
  
   Please download the pretrained models and checkpoints from [GoogleDrive](https://drive.google.com/file/d/1x01wods-LUA6EyAD8C3ahiEaO8lKD6jy/view?usp=sharing).
   
    ```
     unzip NavMorph-8324.zip    
    ```
      Overall, files and folds should be organized as follows:
   
     ```
      NavMorph
      ├── data
      │   ├── checkpoints
      │   │   └── ckpt.pth
      │   ├── vpm_1000_wm_im.pkl
      │   ├── datasets
      │   ├── logs
      │   ├── scene_datasets
      │   └── wp_pred
      │       └── cwp_predictor.pth
      ├── pretrained
      │   ├── NeRF_p16_8x8.pth
      │   ├── ViT-B-32.pt
      │   ├── segm.pt
      │   ├── resnet18-f37072fd.pth
      │   └── model_step_100000.pt
      └── bert_config
          └── bert-base-uncased
     ```

   🧑‍💻 We will soon provide a clean, organized compressed package matching this structure for easy download.

### Training for R2R-CE / RxR-CE

Use pseudo interative demonstrator to train the world model Navmorph:
```
bash run_r2r/main.bash train # (run_rxr/main.bash)
```

### Online Evaluation on R2R-CE / RxR-CE

Use pseudo interative demonstrator to equip the model with our NavMorph:
```
bash run_r2r/main.bash eval # (run_rxr/main.bash)
```

### Notes❗

When transitioning from the R2R dataset to the RxR dataset based on the baseline code, you will need to adjust the camera settings in three places to prevent any simulation issues.

1. **Camera HFOV and VFOV Adjustment**:  
   In [vlnce_bacelines/models/etp/nerf.py](https://github.com/Feliciaxyao/NavMorph/blob/ae3246b902cdedf8533211ff62b2062cb9ed0e39/vlnce_baselines/models/etp/nerf.py#L57-L60), update the camera's **HFOV** and **VFOV**:
   - Set `HFOV = 90` for R2R.
   - Set `HFOV = 79` for RxR.

2. **Dataset Setting**:  
   In [vlnce_bacelines/models/Policy_ViewSelection_ETP.py](https://github.com/Feliciaxyao/NavMorph/blob/ae3246b902cdedf8533211ff62b2062cb9ed0e39/vlnce_baselines/models/Policy_ViewSelection_ETP.py#L41), modify the `DATASET` variable:
   - Set `DATASET = 'R2R'` for R2R.
   - Set `DATASET = 'RxR'` for RxR.

3. **Camera Configuration**:  
   In [vlnce_baselines/ss_trainer_ETP.py](https://github.com/Feliciaxyao/NavMorph/blob/ae3246b902cdedf8533211ff62b2062cb9ed0e39/vlnce_baselines/ss_trainer_ETP.py#L181), ensure the camera configuration is updated:
   - Set `camera.config.HFOV = 90` for R2R.
   - Set `camera.config.HFOV = 79` for RxR.

These adjustments are essential for proper camera calibration and to avoid discrepancies during simulation.

## 📢 TODO list：

-◻️ Checkpoints for RxR-CE release

-◻️ Pre-trained CEM for RxR-CE release

-◻️ Real-world Verification

## Acknowledgements
Our implementations are partially based on [VLN-3DFF](https://github.com/MrZihan/Sim2Real-VLN-3DFF) and [ETPNav](https://github.com/MarSaKi/ETPNav). Thanks to the authors for sharing their code.


## Related Work
* [Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments](https://arxiv.org/pdf/2004.02857)

## 📝 Citation

If you find this project useful in your research, please consider cite:
```
@inproceedings{yao2025navmorph,
  title={NavMorph: A Self-Evolving World Model for Vision-and-Language Navigation in Continuous Environments},
  author={Xuan Yao, Junyu Gao and Changsheng Xu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={},
  year={2025}
} 
