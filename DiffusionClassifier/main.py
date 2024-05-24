import os

import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, \
    f1_score
from torch.utils.data import Dataset, DataLoader

import DiffusionClassifier
from DiffusionModel import Unet, GaussianDiffusion
import torch
from torch.optim import Adam
from torch import Tensor
from datasets import load_dataset
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json

class DataSetTest(Dataset):

    def __init__(self, tensor: Tensor, labels):
        self.tensor = tensor.clone()
        self.labels = labels
    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        img = self.tensor[idx]
        label = self.labels[idx]

        return img, label

if __name__ == '__main__':
    num_classes = 10
    dataset = load_dataset("mnist")
    image_size = 28
    channels = 1
    batch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

    # define function
    def transforms(examples):
        examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
        del examples["image"]
        return examples


    transformed_dataset = dataset.with_transform(transforms)  # .remove_columns("label")
    # create dataloader
    dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=False)
    dataloader_val = DataLoader(transformed_dataset["test"], batch_size=batch_size, shuffle=False)

    model = Unet(
        dim = 64,
        channels=1,
        dim_mults = (1, 2, 4),
        num_classes = num_classes,
        cond_drop_prob = 0.5
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = image_size,
        timesteps = 1000
    ).cuda()

    optimizer = Adam(model.parameters(), lr=5e-4)

    model.load_state_dict(torch.load("./results/model9.pt", torch.device('cuda')))

    # sample_batch = next(iter(dataloader_val))
    # samples = sample_batch['pixel_values']
    # sample_labels = sample_batch['label']

    predictions = np.array([])
    labels = np.array([])
    for step, batch in enumerate(dataloader_val):
        samples = batch['pixel_values']
        labels = np.concatenate((labels,batch['label']))
        predictions = np.concatenate((predictions,
                                      DiffusionClassifier.Classify(
                                          diffusion, samples,
                                          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                          300)))


    confusion = confusion_matrix(labels, predictions)
    ConfusionMatrixDisplay(confusion).plot()
    plt.savefig(f"{os.getcwd()}/Confusion Matrices/Confusion_Test.png")
    plt.show()


    metrics = {
        "Overall Accuracy": accuracy_score(labels, predictions),
        "Precision Score": precision_score(labels, predictions, average='weighted'),
        "Recall Score": recall_score(labels, predictions, average='weighted'),
        "F1 Score": f1_score(labels, predictions, average='weighted')
    }
    json.dump(metrics, open(f'./LGBM Metrics/metrics_64.json', 'w'))