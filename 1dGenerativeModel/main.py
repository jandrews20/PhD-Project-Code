import denoising_diffusion_pytorch
import DataHandler
import torch
import numpy
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataset = DataHandler.Load1DDatasetMalware("C:/Users/40237845/Downloads/ember_2017_2/train_features_1.jsonl")
    print(len(dataset))

