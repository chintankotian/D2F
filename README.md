# D2F -  Description to Face Synthesis 
### A project to explore the use of transfer learning in  GANs to produce photo realistic images of human faces from its description
### Project breakdown
- [ x ] Data preprocessing(effecient way to load/use the entire Celeb-A data onto the main memory)
- [ ] Language Model
- [ ] Implementing the Giant MSG-GANs on the entire LFW dataset
- [ ] Transfer Learning
- [ ] Final Model
- [ ] Accuracy Measure 

### Data preprocessing(effecient way to load/use the entire LFW data onto the main memory)
Requirement of the Data preprocessing code
* Modular
* Should work on atleast 2 levels of folder structure
  * Ex. images/example.png or images/folder1/example.png
* Should output one batchsize of images per training iteration
* Must preprocess the output batch images
* Low overhead
* Images shouldn't be stored in memory before and after the processing of the batch
* Should be Jupyter Notebook/Collab compatible

### Language Model
Requirements
- [ ] Load(from pickle file,json or txt/csv ) ,preprpocess and encode 
- [ ] Must be able to process text using both word level and sentence level embedding
  * Word level embedding Algorithms  - [Glove](https://nlp.stanford.edu/projects/glove/ "Glove webpage"), Word2Vec, FastText
  * Sentence level embedding Algorithms  - [Infersent](https://github.com/facebookresearch/InferSent "Infersent github page"), [Google Universal Encoder](https://tfhub.dev/google/universal-sentence-encoder/4 "TF-hub page of Universal encoder"), [Elmo](https://tfhub.dev/google/elmo/3 "TF-hub page of Elmo")
- [ ]
