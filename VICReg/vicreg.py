'''
Implementation of VICreg for CILOT
'''
from lib2to3.pytree import Base
import torch
import wandb
import os
import numpy as np
import pyrallis
from dataclasses import asdict, dataclass, field
from pydantic import BaseModel
import typing as tp
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


#then change to yaml file using pyrallis.dump
@dataclass
class VICreg(BaseModel):
    """Hyperparameters from authors implementation"""
    run_name: tp.Optional[str] = None
    dataset: tp.Optional[str] = field(default="CIFAR100")
    exp_dir: Path = field(default=Path("/home/m_bobrin/CILOT-Research/VICReg/exp"))
    #Print logs
    log_freq_time: int = field(default=60)
    #Arch to use as backbone
    arch: str = field(default="resnet50")
    batch_size: int = field(default=1024)
    lr: float = field(default=0.2)
    epochs: int = field(default=100)
    weight_decay: float = 1e-6
    #LOSS
    sim_coeff: float = field(default=25.0)
    std_coeff: float = field(default=25.0)
    cov_coeff: float = field(default=1.0)
    #Wandb
    project: str = "VICReg"
    
    def __post_init__(self) -> None:
        wandb.init(asdict(self))
    
@pyrallis.wrap()
def main(cfg: VICreg):
    
    
    
    
if __name__ == "__main__":
    main()



