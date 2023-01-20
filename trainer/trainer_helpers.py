from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
import torch
import torchaudio
import os
from torch.utils.data import Dataset, DataLoader
import pickle
import gc
from tqdm.notebook import tqdm
from IPython.display import clear_output
from trainer.trainer_modelbases import *


def create_model(modeltype='uncond'):
    gc.collect()
    torch.cuda.empty_cache()
    if modeltype == 'uncond':
        model = DiffusionModel(
            net_t=UNetV0,  # The model type used for diffusion (U-Net V0 in this case)
            in_channels=2,  # U-Net: number of input/output (audio) channels
            channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024],  # U-Net: channels at each layer
            factors=[1, 4, 4, 4, 2, 2, 2, 2, 2],  # U-Net: downsampling and upsampling factors at each layer
            items=[1, 2, 2, 2, 2, 2, 2, 4, 4],  # U-Net: number of repeating items at each layer
            attentions=[0, 0, 0, 0, 0, 1, 1, 1, 1],  # U-Net: attention enabled/disabled at each layer
            attention_heads=8,  # U-Net: number of attention heads per attention item
            attention_features=64,  # U-Net: number of attention features per attention item
            diffusion_t=VDiffusion,  # The diffusion method used
            sampler_t=VSampler,  # The diffusion sampler used
        )
    elif modeltype == 'text_cond':
        model = DiffusionModel(
            net_t=UNetV0,  # The model type used for diffusion (U-Net V0 in this case)
            in_channels=2,  # U-Net: number of input/output (audio) channels
            channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024],  # U-Net: channels at each layer
            factors=[1, 4, 4, 4, 2, 2, 2, 2, 2],  # U-Net: downsampling and upsampling factors at each layer
            items=[1, 2, 2, 2, 2, 2, 2, 4, 4],  # U-Net: number of repeating items at each layer
            attentions=[0, 0, 0, 0, 0, 1, 1, 1, 1],  # U-Net: attention enabled/disabled at each layer
            attention_heads=8,  # U-Net: number of attention heads per attention item
            attention_features=64,  # U-Net: number of attention features per attention item
            diffusion_t=VDiffusion,  # The diffusion method used
            sampler_t=VSampler,  # The diffusion sampler used
            use_text_conditioning=True, # U-Net: enables text conditioning (default T5-base)
            use_embedding_cfg=True, # U-Net: enables classifier free guidance
            embedding_max_length=64, # U-Net: text embedding maximum length (default for T5-base)
            embedding_features=768, # U-Net: text mbedding features (default for T5-base)
            cross_attentions=[0, 0, 0, 1, 1, 1, 1, 1, 1], # U-Net: cross-attention enabled/disabled at each layer
        )
    elif modeltype == 'upsampler':
        model = DiffusionUpsampler(
            net_t=UNetV0, # The model type used for diffusion
            upsample_factor=16,  # The upsample factor (e.g. 16 can be used for 3kHz to 48kHz)
            in_channels=2,  # U-Net: number of input/output (audio) channels
            channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024], # U-Net: channels at each layer
            factors=[1, 4, 4, 4, 2, 2, 2, 2, 2], # U-Net: downsampling and upsampling factors at each layer
            items=[1, 2, 2, 2, 2, 2, 2, 4, 4],  # U-Net: number of repeating items at each layer
            diffusion_t=VDiffusion,  # The diffusion method used
            sampler_t=VSampler,  # The diffusion sampler used
        )   

    model.cuda()
    optimizer = torch.optim.Adam(model.parameters())
    return model, optimizer

def create_new_dataset(args):
    waveforms = []
    args.training_audio_path = args.training_audio_path + '/'
    print(f'Getting dataset at {args.training_audio_path}..')
    for file in os.listdir(args.training_audio_path):
        if file.endswith(".wav") or file.endswith(".flac"):
            waveform, samplerate = torchaudio.load(args.training_audio_path + file)
            waveform = waveform.cuda()
            waveforms.append(waveform)
            print(f'Adding {file} to dataset..')
    print(f'Data set created with {len(waveforms)} waveforms.')
    dataset = AudioDataset(waveforms, args.size)
    with open(f'datasets/{args.modelname}_{args.size}.pkl', 'wb') as file:
        pickle.dump(dataset, file)
    return dataset    


def print_cuda_memory():
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')    


def train_model_with_waveforms(model, optimizer, args):
    torch.cuda.set_device(0)
    print_cuda_memory()
    gc.collect()
    torch.cuda.empty_cache()
    if args.dataset == None or not os.path.exists(args.dataset):
        args.dataset = create_new_dataset(args)
    else:
        try:
            with open(args.dataset, 'rb') as f:
                print('Loading dataset from file ' + args.dataset)
                args.dataset = pickle.load(f)
        except FileNotFoundError:
            args.dataset = create_new_dataset(args)

    dataloader = DataLoader(args.dataset, batch_size=args.batch_size, shuffle=True)
    # enumerate epochs
    for epoch in range(args.num_epochs):
        epoch_override = args.epoch_override
        epoch = epoch_override + epoch
        epoch_i = epoch + 1
        prog_bar = tqdm(dataloader, total=len(dataloader))
        for i, waveform in enumerate(prog_bar):
            # Forward pass
            waveform = waveform[:, :, :args.size]
            loss = model(waveform.cuda())
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prog_bar.set_description(f'waveform: {i+1}/{len(dataloader)}, epoch: {epoch_i}/{args.num_epochs + epoch_override}, loss: {loss}')
        clear_output(wait=True)
        prog_bar.close()
        # save every 5 epochs
        if epoch != 0 and epoch % args.save_every == 0:
            save_model(args.modelname, 'models', loss, epoch, model, optimizer, args.sample_rate)
    print(f'Training done.')
    save_model(args.modelname, 'models', loss, epoch_i, model, optimizer, args.sample_rate)
    return model


def save_model(modelname, path, loss, epoch, model, optimizer, sample_rate):
    print('Saving checkpoint and sampling...')
    ckpt_path = os.path.join(path, f'{modelname}_epoch_{epoch}.ckpt')
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, ckpt_path)
    sample_model(model, filename=f'{modelname}_epoch_{epoch}', custom_dir=modelname, sample_rate=sample_rate)
    print('Succesfully saved model to ' + ckpt_path)
    print('Succesfully saved sample to ' + os.path.join(f'samples/{modelname}', f'{modelname}_epoch_{epoch}.wav'))


def sample_model(model, filename="model_sample", num_steps=50, custom_dir=None, sample_rate=48000):
    model.eval()
    # Turn noise into new audio sample with diffusion
    noise = torch.randn(1, 2, 2**18)  # [batch_size, in_channels, length]
    sample = model.sample(noise.cuda(), num_steps=num_steps) # Suggested num_steps 10-100
    sample = torch.squeeze(sample, 0)

    # check custom dir
    if custom_dir is not None:
        custom_dir = 'samples/' + custom_dir
        if not os.path.exists(custom_dir):
            os.mkdir(custom_dir)
        save_path = os.path.join(custom_dir, filename + '.wav')
    else:
        save_path = os.path.join(custom_dir, filename + '.wav')

    # save
    torchaudio.save(save_path, sample.cpu(), sample_rate=sample_rate)


def load_from_checkpoint(ckpt_path, model, optimizer):
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f'Loaded model from checkpoint {ckpt_path} at epoch {epoch}.')
    return model, epoch