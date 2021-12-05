
# Learning with AMIGo: Adversarially Motivated Intrinsic Goals
This repository contains the re-implementation of the paper Learning with AMIGo: Adversarially Motivated Intrinsic GOals. 

<p align="center">
  <img width="604" height="252" src="https://github.com/allepalma/adversarially-motivated-intrinsic-goals/blob/master/resources/amigo.png">
</p>

The repository is organized in the following way:
* The training process is fully implemented in main/main.py, where functions for acting in the environment, training the model and testing pre-trained results are contained 
* The neural network models for the teacher and the student together with a state-embedding network are contained in the models/models.py folder
* Functions utilized in the training loop are contained in utils_folder/utils.py and utils_folder/env_utils.py
* torchbeast/core contains util functions to implement the IMPALA algorithm for fast distributed policy gradient 

# Citation

Our project reimplements AMIGo, which is described in the following paper:

```bib
@article{campero2020learning,
  title={Learning with AMIGo: Adversarially Motivated Intrinsic Goals},
  author={Campero, Andres and Raileanu, Roberta and K{\"u}ttler, Heinrich and Tenenbaum, Joshua B and Rockt{\"a}schel, Tim and Grefenstette, Edward},
  journal={arXiv preprint arXiv:2006.12122},
  year={2020}
}
```

However, some of the experiments leverage the work described in:

```bib
@misc{zhang2020bebold,
      title={BeBold: Exploration Beyond the Boundary of Explored Regions}, 
      author={Tianjun Zhang and Huazhe Xu and Xiaolong Wang and Yi Wu and Kurt Keutzer and Joseph E. Gonzalez and Yuandong Tian},
      year={2020},
      eprint={2012.08621},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

# Installation

```bash
# create a new conda environment
conda create -n amigo python=3.7
conda activate amigo

# install dependencies
git clone git@github.com:facebookresearch/adversarially-motivated-intrinsic-goals.git
cd adversarially-motivated-intrinsic-goals
pip install -r requirements.txt
```

# Running Experiments

## Train AMIGo on MiniGrid

```bash
# Run AMIGo on MiniGrid Environment
OMP_NUM_THREADS=1 python -m monobeast.minigrid.monobeast_amigo --env MiniGrid-KeyCorridorS5R3-v0 \
--num_actors 40 --modify --generator_batch_size 150 --generator_entropy_cost .05 \
--generator_threshold -.5 --total_frames 600000000 --generator_reward_negative -.3 \
--savedir ./experimentMinigrid
```
Please be sure to use --total_frames as in the paper: <br>
6e8 for KeyCorridorS4R3-v0, KeyCorridorS5R3-v0, ObstructedMaze-2Dlhb-v0, ObstructedMaze-1Q-v0 <br>
3e7 for KeyCorridorS3R3 and ObstructedMaze-1Dl-v0

Moreover, the flag ```--disable_checkpoints``` should be only used if the user does not want to save the model parameters


## Test AMIGo on MiniGrid

```bash
# Run AMIGo on MiniGrid Environment
OMP_NUM_THREADS=1 python -m monobeast.minigrid.monobeast_amigo --env trained_amigo_environment --mode test \
--weight_path path_to_saved_weights --record_video --video_path path_to_video.mp4
```

If the flag ```--record_video``` is used, an mp4 video of a random rollout by the trained agent will be produced at the selected path. To record the video, ffmpeg must be installed.

## Train the baselines on MiniGrid
We used an open sourced [implementation](https://github.com/facebookresearch/impact-driven-exploration) of the exploration baselines (i.e. RIDE, RND, ICM, and Count). This code should be pulled in a separate local repository and run within a separate environment.

```bash
# create a new conda environment
conda create -n ride python=3.7
conda activate ride 

# install dependencies
git clone git@github.com:facebookresearch/impact-driven-exploration.git
cd impact-driven-exploration
pip install -r requirements.txt
```

To reproduce the baseline results in the paper, run:
```bash
OMP_NUM_THREADS=1 python -m python main.py --env MiniGrid-ObstructedMaze-1Q-v0 \
--intrinsic_reward_coef 0.01 --entropy_cost 0.0001
```
with the corresponding best values for the `--intrinsic_reward_coef` and `--entropy_cost` reported in the paper for each model. 

Set `--model` to `ride`, `rnd`, `curiosity`, or `count` for RIDE, RND, ICM, or Count, respectively.

Set `--use_fullobs_policy` for using a full view of the environment as input to the policy network. 

Set `--use_fullobs_intrinsic` for using full views of the environment to compute the intrinsic reward. 

The default uses a partial view of the environment for both the policy and the intrinsic reward.

# License

The code in this repository is released under Creative Commons Attribution-NonCommercial 4.0 International License (CC-BY-NC 4.0).
