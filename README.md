# Lightning-ReLoRA
A public implementation of the ReLoRA pretraining method, built on Lightning-AI's Pytorch Lightning suite.
> https://arxiv.org/abs/2307.05695
> 
> https://github.com/Guitaricet/relora/tree/main

This repository is stil under construction.

Pytorch Lightning splits training into two modules, the Lightning Wrapper, and the Lightning Trainer, to hide the boilerplat you don't need, while exposing the loop methods you do need.

ReLoRA is a pretraining method designed around reducing the costs, compute, and time required to pretrain or finetune AI models by leveraging Low Rank Adapters that can be merged into the model weights at a later time.

We attempt to _even further_ optimize this speed and efficiency bonus by supporting Int8 and GPTQ4 quantization methods during pretraining, allowing you to pretrain massive AI models on consumer grade hardware.

### Dev Notes

Classification model works properly. Cold LR warmup not yet implemented, but optimizer reset works, which is the most important feature.

LoRA layers are very sensitive to overfitting. Be gentle when training, and experiment with hyperparameters (especially dropout and learning rate).

The original paper advises that you pretrain a base model as normal for some number of epochs or steps, prior to using ReLoRA for the rest of pretraining.

The final lora layer will be incompatible with the final model output (as the final model is a merged version of that very layer).

Use float32, bf16-mixed, int8, or gptq-4. Avoid 16-mixed for the time being.

### TO-DO:

- [x] Scaffolding
- [x] Integration
- [x] Example Model/Base Model
- [x] Testing - Classification
- [ ] Testing - Language Model
- [ ] CLI
- [ ] Complete
