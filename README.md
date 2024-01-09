# PhD Project Code
Week of 09/01/2024

- Adapted diffusion model to accept 1D vectors
- Flattened MNIST to use as training data to test 1D implementation, successfully trains but the samples are not of high quality. Possibly due to the fact that the vector lacks context of neighbouring pixels. However, the backgrounds of the generated samples are the right black and the model can produce samples that look somewhat like the label.

Week of 05/12/2023

- Trained conditioned diffusion model on MNIST dataset to test
- Attempted to implement conditioning on 1D model but have not been successful yet

Week of 28/11/2023

- Trained LGBM Classifier on Ember Byte Histograms successfully.
- Classification accuracy of 94%, which is significantly higher than the literature specified (68%)
  - Edit: Found that they used a classification threshold to minimise FPR to less than 0.1%, I achieved this with a threshold of 90% but the classification accuracy was still higher than the literature specified, achieving an accuracy of 84%
- Model file saved for later access

Week of 23/10/2023

- 1D Vector Dataset correctly formatted for model consumption. Packaged into (datasetSize * 1 * 256) tensor and normalised between 0 and 1.
- 1D Diffusion Model trained using 1D Vector Dataset of Histograms. Successfully generates new 1D samples. However the samples are still normalised, this could be remedied by taking the min and max values of the dataset and storing them prior to training so that the dataset can be denormalised.
- Image Diffusion Model trained on MNIST dataset as a test. Samples produced are of a high quality and very representative of the training set.
- Attempt at training LGBM model on histograms was unsuccessful as I could not get the data in the right format for training

Week of 17/10/2023

- Initial look at 1D Generative Models
- Got model quickly trained on random data to check how data is input and output from model. (Small number of training steps, just as a test). Successfully takes in 1D vectors and outputs 1D vectors in the same format.
- Got some code put together to take features from Ember (Byte Histogram) and convert them into tensors. Issues getting data to be read in by model (Work-in-progress)
