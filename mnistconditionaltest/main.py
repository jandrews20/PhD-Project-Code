# separate model definition from training loop cell
from torch.optim import Adam
import torch
from DiffusionModel import Unet, p_losses, num_to_groups, train
from torchvision.utils import save_image
from datasets import load_dataset
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torch.utils.data import DataLoader



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

    # define function
    def transforms(examples):
        examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
        del examples["image"]
        return examples

    transformed_dataset = dataset.with_transform(transforms)  # .remove_columns("label")
    # create dataloader
    dataloader = DataLoader(transformed_dataset["train"], batch_size=128, shuffle=True)
    dataloader_val = DataLoader(transformed_dataset["test"], batch_size=128, shuffle=False)

    model = Unet(
        dim=28,
        channels=1,
        dim_mults=(1, 2, 4)
    )
    model.to(device)

    optimizer = Adam(model.parameters(), lr=5e-4)

    train(model, dataloader, dataloader_val, epochs=10, optimizer=optimizer)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
