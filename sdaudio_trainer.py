from trainer.trainer_helpers import *
from types import SimpleNamespace


def get_args():
    return SimpleNamespace(**{
        'modelname': 'dubdon',
        'training_audio_path': 'audio',
        'batch_size': 20,
        'num_epochs': 2500,
        'size': 524288,
        'save_every': 500,
        'sample_rate': 48000,
        'dataset': 'datasets/dubdon_dataset.pkl',
        'custom_ckpt_path': None
    })

modeldata = create_model()
model = modeldata[0]
optimizer = modeldata[1]
args = get_args()

if args.custom_ckpt_path == None:
    model = train_model_with_waveforms(model, optimizer, args)
elif os.path.exists(args.custom_ckpt_path):
    model = load_from_checkpoint(args.custom_ckpt_path, model, optimizer)
