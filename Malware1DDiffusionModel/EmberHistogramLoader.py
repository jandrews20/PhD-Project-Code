import io
import os
import json
import torch
import numpy as np


def LoadDatasetFromFile(filepath):
    histograms = []
    labels = []

    lines = open(filepath)

    for line in lines:
        jsonParse = json.loads(line)

        if jsonParse["label"] == 0 or jsonParse["label"] == 1:
            histograms.append(np.array(jsonParse['histogram'], dtype=np.float64))
            labels.append(jsonParse['label'])
    histograms = np.array(histograms)
    max = np.max(histograms)
    print("Max of the dataset: " + str(max))

    normalised_histograms = histograms / max
    normalised_histograms = (normalised_histograms * 2) - 1
    return torch.Tensor(histograms).unsqueeze(1), labels, max
