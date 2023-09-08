from torch.nn import Module
from torch.optim import Optimizer
from lightning.pytorch.plugins.precision import PrecisionPlugin
from typing import List, Tuple, Any

class Int8Precision(PrecisionPlugin):
  """This exists only to stop Lightning from attempting to use its own precision settings"""
  def __init__(self) -> None:
    super().__init__()
    self.precision = "int8"
    
  def connect(self, model: Module, optimizers: List[Optimizer], lr_schedulers: List[Any]) -> Tuple[Module, List[Optimizer], List[Any]]:
    model.trainer.__dict__["precision"] = "int8"
    return model, optimizers, lr_schedulers
