import torch
import json

#Load 1DDataset into tensor
def Load1DDatasetMalware(filepath):
    file = open(filepath, "r")

    malwareLines = []
    for line in file:
        jsonObject = json.loads(line)

        if jsonObject["label"] == 1:
            malwareLines.append(jsonObject["histogram"])

    return torch.Tensor(malwareLines)