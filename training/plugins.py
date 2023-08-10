from torch.nn import Module
from torch.optim import Optimizer
from lightning.pytorch.plugins.precision import PrecisionPlugin
from peft import prepare_model_for_kbit_training

class Precision(PrecisionPlugin):
  """This exists only to stop Lightning from attempting to use its own precision settings"""
  def __init__(self) -> None:
    super().__init__()
    self.precision = "8"
    def connect(self, model: Module, optimizers: List[Optimizer], lr_schedulers: List[Any]
      ) -> Tuple[Module, List[Optimizer], List[Any]]:

      # Of note, this presumes to use gradient checkpointing.
      model = prepare_model_for_kbit_training(model)
      return model, optimizers, lr_schedulers
