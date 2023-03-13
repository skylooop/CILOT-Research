# Cross Domain IL
First, set environmental variables as shown below. Then, if you are working on headless server (e.g by ssh on cluster), then you need to change ``os.environ["MUJOCO_GL"] = "egl"`` and choose device which supports EGL rendering.
## Environmental variables:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/bobrin_m_s/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```
## Script for running medium expert level (Currently only one domain):
```
python train_iql_ot.py --env_name=hopper-medium-v2 --expert_env_name=hopper-expert-v2 \\
                       --path_to_save_env=path_to_tmp-folder
```
## Script for running cross-domain
```
python train_iql_ot.py --env_name=hopper-medium-v2 --expert_env_name=walker2d-expert-v2
```
## Wandb
[Wandb project page](https://wandb.ai/cilot/projects)