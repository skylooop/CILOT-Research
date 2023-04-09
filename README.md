# Cross Domain IL
First, set environmental variables as shown below. Then, if you are working on headless server (e.g by ssh on cluster), then you need to change ``os.environ["MUJOCO_GL"] = "egl"`` and choose device which supports EGL rendering.
## Environmental variables:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/bobrin_m_s/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

## Running Cross-Domain IL on OpenAI Gym environments for both expert and agent:
```
python train_iql_ot.py --env_name=hopper-medium-v2 --expert_env_name=walker2d-expert-v2
```

## Running Cross-Domain IL on DeepMindControl Suite environments:
```
python train_iql_ot.py --topk=1 --init_dataset_size=0 --dmc_env=True --env_name=cartpole_balance --expoert_env_name=pendulum_swingup
```
## Wandb
[Wandb project page](https://wandb.ai/simmax21/CILOT?workspace=user-simmax21)