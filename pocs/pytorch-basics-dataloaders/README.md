### What is a dataloader?

* Dataset should be decoupled form the model training
* Dataset is a pytorch abstraction that has:
  * samples and labels
* Dataloader wraps dataset with a iterable 