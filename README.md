# Lightning-ReLoRA
A public implementation of the ReLoRA pretraining method, built on Lightning-AI's Pytorch Lightning suite.

This repository is stil under construction.

Pytorch Lightning splits training into two modules, the Lightning Wrapper, and the Lightning Trainer.

The Lightning Wrapper wraps your model inside of a Lightning Object, which handles basic training methods, allowing you to edit them freely.

The Trainer is more closed, and handles things like ddp, training parameters, loop logic, and others.

Lightning attempts to expose the things you need access to, while keeping the general boilerplate hidden, and your functional code succinct.

### TO-DO:

- [x] Scaffolding
- [ ] Integration
- [x] Example Model/Base Model
- [ ] Testing
- [ ] Complete
