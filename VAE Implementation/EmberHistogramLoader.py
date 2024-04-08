import json
import os
import numpy as np
import torch

def LoadUnlabelledDatasetFromFolder(filepath):

    histograms = []

    for samples in os.listdir(filepath):

        lines = open(filepath + "/" + samples)

        for line in lines:
            jsonParse = json.loads(line)

            if jsonParse["label"] == -1:
                array = np.array(jsonParse['histogram'], dtype=np.float64)
                sum = np.sum(array)
                normalised = np.divide(array, sum)
                histograms.append(normalised)
    histograms = np.array(histograms)
    print("Max of the dataset: " + str(max))

    # normalised_histograms = histograms / max
    return torch.Tensor(histograms)