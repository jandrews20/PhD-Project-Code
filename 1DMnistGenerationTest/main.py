# separate model definition from training loop cell
from torch import Tensor
from torch.optim import Adam
import torch
from DiffusionModel import Unet, p_losses, num_to_groups, train
from torchvision.utils import save_image
from datasets import load_dataset
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torch.utils.data import DataLoader, Dataset


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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    dataset = load_dataset("mnist")

    transform = Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

    transform2 = Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),
        transforms.Lambda(torch.flatten),
        transforms.Lambda(lambda q: torch.unsqueeze(q, 0))
    ])

    # define function
    def transforms(examples):
        examples["pixel_values"] = [transform2(image.convert("L")) for image in examples["image"]]
        del examples["image"]
        return examples

    transformed_dataset = dataset.with_transform(transforms)  # .remove_columns("label")
    # create dataloader
    dataloader = DataLoader(transformed_dataset["train"], batch_size=128, shuffle=True)
    dataloader_val = DataLoader(transformed_dataset["test"], batch_size=128, shuffle=False)

    #rand = torch.rand(100, 1, 784)
    #rand2 = torch.zeros(100, 1, 784)
    #rand3 = torch.ones(100, 1, 784)
    #labels = [0 for i in range(0,100)]
    #print(len(labels))

    #test = DataSetTest(rand, labels)
    #test2 = DataSetTest(rand, labels)
   # d1 = DataLoader(test, batch_size=4, shuffle=False, num_workers=4)
    #d2 = DataLoader(test, batch_size=4, shuffle=False, num_workers=4)

    #for step, batch in enumerate(d1):
    #    print(batch[0])

    model = Unet(
        dim=28,
        channels=1,
        dim_mults=(1, 2, 4)
    )
    model.to(device)

    optimizer = Adam(model.parameters(), lr=5e-4)

    train(model, dataloader, dataloader_val, epochs=50, optimizer=optimizer)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
