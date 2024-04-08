import json
import io
import os

import numpy as np
import pandas as pd

def DataSetLoadSingleFile(filepath):
    lines = open(filepath)

    histogramArray = []
    labelArray = []
    for line in lines:
        jsonParse = json.loads(line)

        if jsonParse["label"] == 0 or jsonParse["label"] == 1:
            histogram = jsonParse["histogram"]
            sum = np.sum(histogram)
            normalised_histogram = histogram / sum
            histogramArray.append(normalised_histogram)
            labelArray.append(jsonParse["label"])

    df = pd.DataFrame(histogramArray)
    df = df.assign(Label=labelArray)
    return df

def DataSetLoadFolder(filepath):

    histogramArray = []
    labelArray = []

    for samples in os.listdir(filepath):

        lines = open(filepath + "/" + samples)

        for line in lines:
            jsonParse = json.loads(line)

            if jsonParse["label"] == 0 or jsonParse["label"] == 1:
                histogram = jsonParse["histogram"]
                sum = np.sum(histogram)
                normalised_histogram = histogram / sum
                histogramArray.append(normalised_histogram)
                labelArray.append(jsonParse["label"])

    df = pd.DataFrame(histogramArray)
    df = df.assign(Label=labelArray)
    return df

