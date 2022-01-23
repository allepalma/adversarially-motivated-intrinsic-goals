
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
git clone https://github.com/allepalma/adversarially-motivated-intrinsic-goals.git
cd adversarially-motivated-intrinsic-goals
pip install -r requirements.txt
```

# Running Experiments

Three main experiments can be reproduced via the main/main.py script:
* The default AMIGo run
* AMIGo with adaptive t* threshold
* An AMIGo model where the intrinsic reward function is substituted with one derived from random network distillation 

## Train AMIGo on MiniGrid

```bash
# Run AMIGo on MiniGrid Environment
OMP_NUM_THREADS=1 python -m main.main --env MiniGrid-KeyCorridorS3R3-v0 \
--num_actors 40 --modify --generator_batch_size 150 --generator_entropy_cost .05 \
--generator_threshold -.1 --total_frames 100000000 --generator_reward_negative -.3 \
--savedir ./experimentMinigrid --model default
```

In our project, we mostly worked on the MiniGrid-KeyCorridorS3R3-v0 envronment. However, the main.py script can be run
on any other available MiniGrid envoronment as well. The parameters here are the dafault ones from the paper.

## Train AMIGo with adaptive t* threshold

```bash
# Run AMIGo on MiniGrid Environment
OMP_NUM_THREADS=1 python -m main.main --env MiniGrid-KeyCorridorS3R3-v0 \
--num_actors 40 --modify --generator_batch_size 150 --generator_entropy_cost .05 \
--generator_threshold -.5 --total_frames 100000000 --generator_reward_negative -.3 \
--savedir ./experimentMinigrid --model window_addaptive --window 50
```
The --window flag can be changed to any preferred value. 

## Train AMIGo with BeBold IR function

```bash
# Run AMIGo on MiniGrid Environment
OMP_NUM_THREADS=1 python -m main.main --env MiniGrid-KeyCorridorS3R3-v0 \
--num_actors 10 --modify --total_frames 100000000 \
--savedir ./experimentMinigrid --model novelty_based
```

Less actors are employed since it yielded better learning outcomes. 

## Test AMIGo on MiniGrid

```bash
# Run AMIGo on MiniGrid Environment
OMP_NUM_THREADS=1 python -m main.main --env trained_amigo_environment  \
--mode test --weight_path path_to_saved_weights --record_video --video_path path_to_video.mp4
```

The flag ```--model``` followed by default, window_adaptive and novelty_based must as well be specified.

If the flag ```--record_video``` is used, an mp4 video of a random rollout by the trained agent will be produced at the selected path. To record the video, ffmpeg must be installed.



# License

The code in this repository is released under Creative Commons Attribution-NonCommercial 4.0 International License (CC-BY-NC 4.0).
