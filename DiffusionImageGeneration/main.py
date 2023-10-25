from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim=56,
    dim_mults=(1, 2, 4,),
    channels=1,
    flash_attn=False
)

diffusion = GaussianDiffusion(
    model,
    image_size=28,
    timesteps=1000,
    sampling_timesteps=250
)

trainer = Trainer(
    diffusion,
    './FullTestData',
    train_batch_size=32,
    save_and_sample_every=1000,
    train_lr=8e-5,
    train_num_steps=70000,  # total training steps
    gradient_accumulate_every=2,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    amp=False,  # turn on mixed precision
    calculate_fid=False # whether to calculate fid during training
)

trainer.train()
