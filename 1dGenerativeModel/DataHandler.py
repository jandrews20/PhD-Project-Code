import torch
import json
from sklearn import preprocessing
import numpy as np

#Load 1DDataset into tensor
def Load1DDatasetMalware(filepath):
    file = open(filepath, "r")

    malwareLines = []
    for line in file:
        jsonObject = json.loads(line)

        if jsonObject["label"] == 1:
            malwareLines.append(jsonObject["histogram"])

        if len(malwareLines) == 100:
            break
    malwareLines = preprocessing.normalize(malwareLines)
    return torch.Tensor(malwareLines).unsqueeze(1).float()