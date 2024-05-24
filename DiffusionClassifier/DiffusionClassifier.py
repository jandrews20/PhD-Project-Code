import collections

import numpy as np
import torch
import random

from matplotlib import pyplot as plt
from tqdm import tqdm

import torch.nn.functional as F

@torch.no_grad()
def Classify(model, batch, classes, numTrials):
    errors = collections.defaultdict(dict)

    for i in range(batch.shape[0]):
        for j in range(len(classes)):
            errors[i][j] = list()

    # Add back channel dimension if needed
    if batch.dim() == 3:
        batch = batch.unsqueeze(1)

    for i in tqdm(range(numTrials)):
        # sample t ~ [1,1000]; e ~ N(0,1)
        newTensor = torch.clone(batch).cuda()
        noise = torch.randn_like(batch[0]).cuda()
        t = torch.tensor(np.repeat(random.randint(0, 999), batch.shape[0]), dtype=torch.int64).cuda()

        # Xt = sqrt(a_bar_t)x + sqrt(1-a_bar_t)e
        x_t = model.q_sample(newTensor.cuda(), t=t, noise=noise).cuda()

        # Sample noise prediction for each class and calculate error
        for j in range(len(classes)):
            #model_out = model.model_predictions(x_t, t, torch.IntTensor(np.repeat(j, batch.shape[0])).cuda())
            model_out = model.model(x_t, t, classes=torch.IntTensor(np.repeat(j, batch.shape[0])).cuda())
            for sample in range(batch.shape[0]):
                #error = np.mean(np.abs((noise.detach().cpu().numpy() - model_out[0][sample].cpu().detach().numpy())) ** 2)
                error = np.mean(np.abs((noise.detach().cpu().numpy() - model_out[sample].cpu().detach().numpy())) ** 2)
                errors[sample][j].append(error)

    # Take mean of errors for each class and take argmin to get the classification
    classLabels = []
    for key in errors.keys():
        prediction_errors = []
        for label in classes:
            prediction_errors.append(np.mean(errors[key][label]))
        classLabels.append(np.argmin(prediction_errors))
    return classLabels

