"""
Basic trainer template for multi-gpu, just so we have scaffolding to work from.
Lifted from -> https://github.com/ibeltagy/pytorch-lightning/blob/master/pl_examples/basic_examples/gpu_template.py
"""
import os
from argparse import ArgumentParser
from transformers import LlamaForCausalLM
from peft import LoraConfig

from pytorch_lightning import Trainer, seed_everything
from training.lightning_wrapper import ReloraModule, lora_modules

seed_everything(234)


def main(args):
    """ Main training routine specific for this project. """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b",
                                             load_in_8bit=True,)
    model = ReloraModule(model=model,
                         lora_config=LoraConfig(**vars(args)),
                        **vars(args))

    #Elle's notes - ex:
    #model = model.from_pretrained(path)
    # -- OR --
    #model = model(config).init_weights()
    #-- THEN --
    """model = LightningModel(model=model,
                            optimizer=optimizer,
                            etc)"""

    # Begin training
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)

#Elle's note - I'm unsure if CLI will work with "Bring your own model". We could either build it on top of Llama or GPTnano or something,
# or keep it hackable and general. Or both.
def run_cli():
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)

    # Establish your custom CLI variables in the module itself
    parser = ReloraModule.add_model_specific_args(parent_parser, root_dir)
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(gpus=2)
    parser.set_defaults()

    # Initializing args
    args = parser.parse_args()
    args.lora_target_modules = lora_modules[args.lora_modules] #Set lora_modules from our lookup dictionary

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(args)


if __name__ == '__main__':
    run_cli()
