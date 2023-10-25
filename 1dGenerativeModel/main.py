from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
import DataHandler
import torch
import numpy
# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    model = Unet1D(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=1
    )

    diffusion = GaussianDiffusion1D(
        model,
        seq_length=256,
        timesteps=1000,
        objective='pred_v'
    )

    training_seq = DataHandler.Load1DDatasetMalware("./Dataset/train_features_1.jsonl")
    dataset = Dataset1D(training_seq)

    print(training_seq)

    trainer = Trainer1D(
        diffusion,
        dataset=dataset,
        save_and_sample_every=100,
        train_batch_size=32,
        train_lr=8e-5,
        train_num_steps=7000,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=True,  # turn on mixed precision
    )
    #trainer.train()


