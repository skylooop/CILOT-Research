import os
from abc import abstractmethod, ABC
from tensorboardX import SummaryWriter

import wandb
import typing as tp

class InitLogging(ABC):
    
    @abstractmethod
    def init(self, save_dir: str, seed: int, wandb_project_name: tp.Optional[str], 
             wandb_entity: tp.Optional[str], wandb_job_type: tp.Optional[str]) -> None:
        pass
    
    
class InitTensorboard(InitLogging):
    def init(self, save_dir: str, seed: int) -> SummaryWriter:
        os.makedirs(save_dir, exist_ok=True)
        
        summary_writer = SummaryWriter(
            os.path.join(save_dir, "tb", str(seed)), write_to_disk=True
        )
        return summary_writer
    
#class InitWandb(InitLogging):
    #def init(self, save_dir: str, seed: int) -> :
        #run = wandb.init(project)