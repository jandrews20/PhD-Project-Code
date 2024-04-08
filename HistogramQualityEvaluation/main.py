import numpy as np
from torch import Tensor
from torch.optim import Adam
import torch
import EmberHistogramLoader
from DiffusionModel import Unet, p_losses, num_to_groups, train, sample
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from MalwareClassEvaluator import EvaluateMalwareClassQualityBinary, EvaluateMalwareClassQualityMulticlass
import matplotlib.pyplot as plt
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

def denormalize_neg_one_to_zero(data):
    return (data + 1) * 0.5

def denormalize_zero_to_one(data, max):
    return data * max

def round_to_neg_one(data):
    data = np.where(data < -1, -1, data)
    return data

def createImage(sampleList, label):
    sampleList = sampleList[-1].reshape(256,256)
    sampleList = sampleList * 255
    sampleList = sampleList.astype('uint8')
    test = Image.fromarray(sampleList)
    test.save(os.getcwd() + f'./image{"Benign" if label == 0 else "Malware"}.png')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    #dataset = load_dataset("mnist")


    #histograms, labels, maxValue = EmberHistogramLoader.LoadDatasetFromFile("C:/Users/40237845/Documents/Ember_2017/ember_2017_2/test_features.jsonl")
    #hist_train, hist_test, label_train, label_test = train_test_split(histograms, labels, test_size = 0.3, random_state = 4)


    #emberDataset = DataSetTest(hist_train, label_train)
    #testEmberDataset = DataSetTest(hist_test, label_test)

    #emberDataloader = DataLoader(emberDataset, batch_size=128, shuffle=True)
    #testEmberDataloader = DataLoader(testEmberDataset, batch_size=128, shuffle=True)

    model = Unet(
        dim=128,
        channels=1,
        dim_mults=(1, 2, 4)
    )
    model.to(device)

    optimizer = Adam(model.parameters(), lr=5e-4)

    #train(model, emberDataloader, testEmberDataloader, epochs=100, optimizer=optimizer)

    model.load_state_dict(torch.load("./resultsEmberTrainData3Class/model99.pt", torch.device('cpu')))
    #LGBMmodel = joblib.load("MalwareFamilyLGBMV2.pkl")

    malwareClasses = ["xtrat", "installmonster", "zusy", "vtflooder", "zbot", "fareit", "ramnit", "salty", "adposhel",
                      "emotet"]
    #EvaluateMalwareClassQualityBinary(model, LGBMmodel, malwareClasses)
    #EvaluateMalwareClassQualityMulticlass(model, LGBMmodel, malwareClasses=malwareClasses)

    label = 0
    samplesBenign = np.array(sample(model, label, 256, 1000, 1))
    samplesBenign = round_to_neg_one(samplesBenign)
    samplesBenign = denormalize_neg_one_to_zero(samplesBenign)
    samplesBenign = np.squeeze(samplesBenign, axis=2)
    #
    label = 1
    samplesMalware = np.array(sample(model, label, 256, 1000, 1))
    samplesMalware = round_to_neg_one(samplesMalware)
    samplesMalware = denormalize_neg_one_to_zero(samplesMalware)
    samplesMalware = np.squeeze(samplesMalware, axis=2)
    #
    joinedSamples = np.concatenate((samplesBenign[-1], samplesMalware[-1]))
    #
    LGBMmodel = joblib.load("lgb.pkl")
    test_predBenign = LGBMmodel.predict(samplesBenign[-1])
    print(test_predBenign)
    print(np.count_nonzero(test_predBenign == 0))
    print(f'Test Benign Accuracy: {np.count_nonzero(test_predBenign == 0)/1000}')
    labels = np.concatenate((np.zeros(1000),np.ones(1000)))
    test_predMalware = LGBMmodel.predict(samplesMalware[-1])
    #print(test_predMalware)
    print(np.count_nonzero(test_predMalware == 1))
    print(f'Test Malware Accuracy: {np.count_nonzero(test_predMalware == 1) / 1000}')
    # #samples = denormalize_zero_to_one(samples, maxValue)
    #
    test_predict = LGBMmodel.predict(joinedSamples)
    print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(labels, test_predict)))
    confusion = confusion_matrix(labels, test_predict)
    ConfusionMatrixDisplay(confusion, display_labels=["Benign", "Malware"]).plot(xticks_rotation="vertical")
    plt.show()
    print('Confusion matrix\n\n', confusion)
    print('\nTrue Positives(TP) = ', confusion[0, 0])
    print('\nTrue Negatives(TN) = ', confusion[1, 1])
    print('\nFalse Positives(FP) = ', confusion[0, 1])
    print('\nFalse Negatives(FN) = ', confusion[1, 0])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
