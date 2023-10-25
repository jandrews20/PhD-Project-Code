# PhD Project Code

Week of 23/10/2023

- 1D Vector Dataset correctly formatted for model consumption. Packaged into (datasetSize * 1 * 256) tensor and normalised between 0 and 1.
- 1D Diffusion Model trained using 1D Vector Dataset of Histograms. Successfully generates new 1D samples. However the samples are still normalised, this could be remedied by taking the min and max values of the dataset and storing them prior to training so that the dataset can be denormalised.
- Image Diffusion Model trained on MNIST dataset as a test. Samples produced are of a high quality and very representative of the training set.
- Attempt at training LGBM model on histograms was unsuccessful as I could not get the data in the right format for training


Week of 17/10/2023

- Initial look at 1D Generative Models
- Got model quickly trained on random data to check how data is input and output from model. (Small number of training steps, just as a test). Successfully takes in 1D vectors and outputs 1D vectors in the same format.
- Got some code put together to take features from Ember (Byte Histogram) and convert them into tensors. Issues getting data to be read in by model (Work-in-progress)
