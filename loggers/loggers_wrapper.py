import os
from abc import abstractmethod, ABC
from tensorboardX import SummaryWriter
import typing as tp
from wandb import sdk as wandb_sdk

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
    
class InitWandb(InitLogging):
    def init(self, config, save_dir: str, seed: int, wandb_project_name: tp.Optional[str], 
             wandb_entity: tp.Optional[str], wandb_job_type: tp.Optional[str]) -> wandb_sdk.wandb_run.Run:
        import wandb
        
        wandb.login()
        run = wandb.init(config=config, project=wandb_project_name, entity=wandb_entity,
                         job_type=wandb_job_type, tags=["One Domain IL"])
        
        return run