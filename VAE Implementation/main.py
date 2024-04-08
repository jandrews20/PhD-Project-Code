import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import Tensor
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.optim import Adam
from tqdm import tqdm
import EmberHistogramLoader
from sklearn.model_selection import train_test_split
import sys
import json
class DataSetTest(Dataset):

    def __init__(self, tensor: Tensor):
        self.tensor = tensor.clone()
    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        img = self.tensor[idx]

        return img

if __name__ == '__main__':
    cuda = True
    DEVICE = torch.device("cuda" if cuda else "cpu")

    dataset = EmberHistogramLoader.LoadUnlabelledDatasetFromFolder("C:/Users/40237845/Documents/Ember_2017/ember_2017_2/train")
    train_data, test_data = train_test_split(dataset, test_size=0.1, random_state=42)

    dataset_train = DataSetTest(train_data)
    dataset_test = DataSetTest(test_data)
    batch_size = 256

    lr = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-3
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    latent_dim = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    print(lr)
    x_dim = 256
    hidden_dim = 200

    kwargs = {'num_workers': 1, 'pin_memory': True}

    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, **kwargs)


    class Encoder(nn.Module):

        def __init__(self, input_dim, hidden_dim, latent_dim):
            super(Encoder, self).__init__()

            self.FC_input = nn.Linear(input_dim, hidden_dim)
            self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
            self.FC_mean = nn.Linear(hidden_dim, latent_dim)
            self.FC_var = nn.Linear(hidden_dim, latent_dim)

            self.LeakyReLU = nn.LeakyReLU(0.2)

            self.training = True

        def forward(self, x):
            h_ = self.LeakyReLU(self.FC_input(x))
            h_ = self.LeakyReLU(self.FC_input2(h_))
            mean = self.FC_mean(h_)
            log_var = self.FC_var(h_)  # encoder produces mean and log of variance
            #             (i.e., parameters of simple tractable normal distribution "q"

            return mean, log_var


    class Decoder(nn.Module):
        def __init__(self, latent_dim, hidden_dim, output_dim):
            super(Decoder, self).__init__()
            self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
            self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
            self.FC_output = nn.Linear(hidden_dim, output_dim)

            self.LeakyReLU = nn.LeakyReLU(0.2)

        def forward(self, x):
            h = self.LeakyReLU(self.FC_hidden(x))
            h = self.LeakyReLU(self.FC_hidden2(h))

            x_hat = torch.sigmoid(self.FC_output(h))
            return x_hat


    class Model(nn.Module):
        def __init__(self, Encoder, Decoder):
            super(Model, self).__init__()
            self.Encoder = Encoder
            self.Decoder = Decoder

        def reparameterization(self, mean, var):
            epsilon = torch.randn_like(var).to(DEVICE)  # sampling epsilon
            z = mean + var * epsilon  # reparameterization trick
            return z

        def forward(self, x):
            mean, log_var = self.Encoder(x)
            z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
            x_hat = self.Decoder(z)

            return x_hat, mean, log_var


    encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)

    model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)

    BCE_loss = nn.BCELoss()


    def loss_function(x, x_hat, mean, log_var):
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return reproduction_loss + KLD


    optimizer = Adam(model.parameters(), lr=lr)

    print("Start training VAE...")
    model.train()
    def train():

        loss_history = {}

        for epoch in range(epochs):
            overall_loss = 0
            for batch_idx, (x) in enumerate(train_loader):
                x = x.to(DEVICE)

                optimizer.zero_grad()

                x_hat, mean, log_var = model(x)
                loss = loss_function(x, x_hat, mean, log_var)

                overall_loss += loss.item()

                loss.backward()
                optimizer.step()

            print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx * batch_size))
            loss_history[epoch] = {"average_train_loss": overall_loss / (batch_idx * batch_size)}
        json.dump(loss_history, open(f"./Loss Histories/loss_history_{lr}_{latent_dim}.json", "w"))


    train()

    torch.save(model.state_dict(), f"./results/MalwareVAEmodel_{lr}_{latent_dim}.pt")
    #model.load_state_dict(torch.load("./MalwareVAEmodel.pt", torch.device('cpu')))

    #model.eval()

    #with torch.no_grad():
    #    for batch_idx, x in enumerate(tqdm(test_loader)):
    #        x = x.to(DEVICE)

    #        x_hat, _, _ = model(x)

            #print(x[0])
            #print(x_hat[0])
            #print(x[-1])
            #print(x_hat[-1])
    #        break
