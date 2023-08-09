import torch
import lightning as L
#All code borrowed or modified from https://github.com/Guitaricet/relora

"""
These are my initial methods from my Fabric trainer. We'll need to port them into the full Lightning trainer, either in the wrapper, or through
plugins/callbacks
"""

# Of note, Fabric and Lightning have their own plugins. We want to use -> from lightning.pytorch.plugins.precision import PrecisionPlugin
# for our version
from lightning.fabric.plugins import Precision
class Precision(Precision):
    """This exists only to stop Lightning from attempting to use its own precision settings"""
    precision: "8"
    def __init__(self):
        super().__init__()

#These are pulled from my trainer class. We'll have to think about how to implement them into the checkpointer.
def reset_optimizer(self, optimizer):
  """This was part of the trainer originally, references may need to be changed, since we now only has have
  access to the wrapper and the initial training args"""
  self.fabric.logger.info("Resetting optimizer states to zeros")

  #This assumes that optimizer is not wrapped by accelerator
  for group in optimizer.param_groups:
      for p in group["params"]:
          param_state = optimizer.state[p]
          param_state["exp_avg"] = torch.zeros_like(p.data)
          param_state["exp_avg_sq"] = torch.zeros_like(p.data)
  #Of note, ReLoRA has two "settings", one just saves and loads the optimizer again, instead of setting all values to 0.
  return optimizer

def merge_and_reinit(self, model, optimizer)
  model = model.merge_and_unload()
  model = PeftModel(model, lora_config)
  #Accelerator will lose track of NAN checks right here. I had to rewrap the optimizer to fix this.
  model, optimizer = accelerator.apply(model, optimizer)
  
  return model, optimizer
