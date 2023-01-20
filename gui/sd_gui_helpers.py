from audio_diffusion_pytorch import DiffusionUpsampler, DiffusionModel, UNetV0, VDiffusion, VSampler
import torch
import torchaudio
import sys


def get_upsampler():
    return DiffusionUpsampler(
        net_t=UNetV0, # The model type used for diffusion
        upsample_factor=16, # The upsample factor (e.g. 16 can be used for 3kHz to 48kHz)
        in_channels=2, # U-Net: number of input/output (audio) channels
        channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024], # U-Net: channels at each layer
        factors=[1, 4, 4, 4, 2, 2, 2, 2, 2], # U-Net: downsampling and upsampling factors at each layer
        items=[1, 2, 2, 2, 2, 2, 2, 4, 4], # U-Net: number of repeating items at each layer
        diffusion_t=VDiffusion, # The diffusion method used
        sampler_t=VSampler, # The diffusion sampler used
    )


def train_upsampler(upsampler=get_upsampler()):
    # Train model with high sample rate audio waveforms
    audio = torch.randn(1, 2, 2**18) # [batch, in_channels, length]
    loss = upsampler(audio)
    loss.backward()


def upsample_audio(upsampler):
    downsampled_audio = torch.randn(1, 2, 2**14) # [batch, in_channels, length]
    sample = upsampler.sample(downsampled_audio, num_steps=10) # Output has shape: [1, 2, 2**18]


def generate_audio(values):
    print("Getting prompts.")
    prompts = get_prompts(values)
    print('using prompts: ', prompts)
    print("Getting model.")
    model = get_model()
    print("Generating audio.")
    generate_audio_from_model(prompts, model=model)
    print("Done.")

def get_model():
    return DiffusionModel(
        net_t=UNetV0,  # The model type used for diffusion (U-Net V0 in this case)
        in_channels=2,  # U-Net: number of input/output (audio) channels
        channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024],  # U-Net: channels at each layer
        factors=[1, 4, 4, 4, 2, 2, 2, 2, 2],  # U-Net: downsampling and upsampling factors at each layer
        items=[1, 2, 2, 2, 2, 2, 2, 4, 4],  # U-Net: number of repeating items at each layer
        attentions=[0, 0, 0, 0, 0, 1, 1, 1, 1],  # U-Net: attention enabled/disabled at each layer
        attention_heads=8, # U-Net: number of attention heads per attention item
        attention_features=64,  # U-Net: number of attention features per attention item
        diffusion_t=VDiffusion, # The diffusion method used
        sampler_t=VSampler,  # The diffusion sampler used
        use_text_conditioning=True,  # U-Net: enables text conditioning (default T5-base)
        use_embedding_cfg=True,  # U-Net: enables classifier free guidance
        embedding_max_length=64,  # U-Net: text embedding maximum length (default for T5-base)
        embedding_features=768,  # U-Net: text mbedding features (default for T5-base)
        cross_attentions=[0, 0, 0, 1, 1, 1, 1, 1, 1],  # U-Net: cross-attention enabled/disabled at each layer
    )


def generate_audio_from_model(prompts, model=get_model(), embedding_scale=5.0, num_steps=50, length=2**18):
    audio_wave = torch.randn(len(prompts), 2, length)  # [batch, in_channels, length]
    loss = model(
        audio_wave,
        text=prompts,  # Text conditioning, one element per batch
        embedding_mask_proba=0.1  # Probability of masking text with learned embedding (Classifier-Free Guidance Mask)
    )
    loss.backward()
    noise = torch.randn(len(prompts), 2, length)
    sample = model.sample(
        noise,
        text=prompts,
        embedding_scale=embedding_scale, # Higher for more text importance, suggested range: 1-15 (Classifier-Free Guidance Scale)
        num_steps=num_steps # Higher for better quality, suggested num_steps: 10-100
    )
    sample = torch.squeeze(sample, 0)
    torchaudio.save("sample_denoise.wav", sample, sample_rate=44100, format="WAV")



def get_prompts(values):
    prompts = values['-PROMPTS-'].split('\n')
    prompts = [p for p in prompts if p]
    return prompts