import matplotlib.pyplot as plt
import numpy as np
def ExtractLossValues(losses):
    train_loss = []
    validation_loss = []

    for key in losses.keys():
        train_loss.append(losses[key]["train_loss"])
        validation_loss.append(losses[key]["validation_loss"])

    return train_loss, validation_loss
def PlotLosses(losses, epochs, dim):
    train_loss, validation_loss = ExtractLossValues(losses)

    epochsRange = range(1, epochs + 1)

    plt.plot(epochsRange, train_loss, label='Training Loss')
    plt.plot(epochsRange, validation_loss, label='Validation Loss')

    plt.title(f'Training and Validation Loss (Dim = {dim})')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')

    plt.xticks(np.arange(0, epochs + 1, 2))

    plt.legend(loc='best')
    plt.show()