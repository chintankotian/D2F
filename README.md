# D2F
## A project to explore the use of transfer learning in  GANs to produce photo realistic images of human faces from its description
### Project breakdown
- [ ] Data preprocessing(effecient way to load/use the entire LFW data onto the main memory)
- [ ] Language Model
- [ ] Implementing the Giant MSG-GANs on the entire LFW dataset
- [ ] Transfer Learning
- [ ] Final Model
- [ ] Accuracy Measure 

#### Data preprocessing(effecient way to load/use the entire LFW data onto the main memory)
Requirement of the Data preprocessing code
* Modular
* Should work on atleast 2 levels of folder structure
  * Ex. images/example.png or images/folder1/example.png
* Should output one batchsize of images per training iteration
* Must preprocess the output batch images
* Low overhead
* Images shouldn't be stored in memory before and after the processing of the batch
* Should be Jupyter Notebook/Collab compatible
