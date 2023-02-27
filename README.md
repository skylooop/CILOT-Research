# Contrastive Cross-Domain IL

## Script for running medium expert level (Currently only one domain):

```
python train_iql_ot.py --env_name=hopper-medium-v2 --expert_env_name=hopper-expert-v2 \\
                       --path_to_save_env=path_to_tmp-folder
```
## Script for running cross-domain
```
python train_iql_ot.py --env_name=hopper-medium-v2 --expert_env_name=walker2d-expert-v2
```

# Requirements:
`mujoco-2.1.0`

## Wandb
[Wandb project page](https://wandb.ai/cilot/projects)