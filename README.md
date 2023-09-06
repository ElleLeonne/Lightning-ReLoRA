# Lightning-ReLoRA
A public implementation of the ReLoRA pretraining method, built on Lightning-AI's Pytorch Lightning suite.
> https://arxiv.org/abs/2307.05695
> 
> https://github.com/Guitaricet/relora/tree/main

This repository is stil under construction.

Pytorch Lightning splits training into two modules, the Lightning Wrapper, and the Lightning Trainer.

The Lightning Wrapper wraps your model inside of a Lightning Object, which handles basic training methods, allowing you to edit them freely.

The Trainer is more closed, and handles things like ddp, training parameters, loop logic, and others.

Lightning attempts to expose the things you need access to, while keeping the general boilerplate hidden, and your functional code succinct.

### Dev Notes
Classification model works properly. Cold LR warmup not yet implemented, but optimizer reset works, which is the most important feature.

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
