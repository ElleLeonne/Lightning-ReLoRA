"""
Basic trainer template for multi-gpu, just so we have scaffolding to work from.
Lifted from -> https://github.com/ibeltagy/pytorch-lightning/blob/master/pl_examples/basic_examples/gpu_template.py
"""
import os
from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from training.lightning_wrapper import LightningModel

seed_everything(234)


def main(args):
    """ Main training routine specific for this project. """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = LightningModel(**vars(args))

    #Elle's notes - ex:
    #model = model.from_pretrained(path)
    # -- OR --
    #model = model(config).init_weights()
    #-- THEN --
    """model = LightningModel(model=model,
                            optimizer=optimizer,
                            etc)"""

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = Trainer.from_argparse_args(args)

    # ------------------------
    # 3 START TRAINING
    # ------------------------
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

    # each LightningModule defines arguments relevant to it
    parser = LightningTemplateModel.add_model_specific_args(parent_parser, root_dir)
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(gpus=2)
    args = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(args)


if __name__ == '__main__':
    run_cli()
