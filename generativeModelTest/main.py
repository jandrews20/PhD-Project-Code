import torch
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
if __name__ == '__main__':
    model = Unet1D(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels = 32
    )

    diffusion = GaussianDiffusion1D(
        model,
        seq_length = 128,
        timesteps = 1000,
        objective = 'pred_v'
    )

    training_seq = torch.rand(64, 32, 256)
    dataset = Dataset1D(training_seq)

    trainer = Trainer1D(
        diffusion,
        dataset = dataset,
        train_batch_size = 32,
        train_lr = 8e-5,
        train_num_steps = 7000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = False,                       # turn on mixed precision
    )
    #trainer.train()

    trainer.load(1)
    sample = diffusion.sample(1)
    print(sample)
