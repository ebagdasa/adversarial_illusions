from argparse import Namespace
from pathlib import Path
import toml
from dataset_utils import create_dataset, get_embeddings
from models import load_model
import librosa
import scipy.signal
import torch
import numpy as np
from DiffJPEG.compression import compress_jpeg
from DiffJPEG.decompression import decompress_jpeg
from DiffJPEG.jpeg_utils import diff_round, quality_to_factor


criterion = torch.nn.functional.cosine_similarity
EMBEDDING_FNM = 'outputs/embeddings/{}_{}_embeddings.npy'
IMG_MEAN=(0.48145466, 0.4578275, 0.40821073)
IMG_STD=(0.26862954, 0.26130258, 0.27577711)
THERMAL_MEAN=(0.2989 * IMG_MEAN[0]) + (0.5870 * IMG_MEAN[1]) + (0.1140 * IMG_MEAN[2])
THERMAL_STD=(0.2989 * IMG_STD[0]) + (0.5870 * IMG_STD[1]) + (0.1140 * IMG_STD[2])
# ImageBind's audio preprocessing stats, see imagebind.data.load_and_transform_audio_data
AUDIO_MEAN=-4.268
AUDIO_STD=9.138

def unnorm(tensor, mean=IMG_MEAN, std=IMG_STD):
    device = tensor.get_device() if tensor.get_device() > 0 else 'cpu'
    if type(mean) != float:
        mean = torch.tensor(mean)[None, :, None, None]
        std = torch.tensor(std)[None, :, None, None]
    return ((tensor.clone().cpu() * std) + mean).to(device)

def norm(tensor, mean=IMG_MEAN, std=IMG_STD):
    device = tensor.get_device() if tensor.get_device() > 0 else 'cpu'
    if type(mean) != float:
        mean = torch.tensor(mean)[None, :, None, None]
        std = torch.tensor(std)[None, :, None, None]
    return ((tensor.clone().cpu() - mean) / std).to(device)


def unnorm_audio(tensor, mean=AUDIO_MEAN, std=AUDIO_STD):
    device = tensor.get_device() if tensor.get_device() > 0 else 'cpu'
    m = torch.tensor(AUDIO_MEAN)
    s = torch.tensor(AUDIO_STD)
    return ((tensor.clone().cpu() * s) + m).to(device)

def norm_audio(tensor, mean=AUDIO_MEAN, std=AUDIO_STD):
    device = tensor.get_device() if tensor.get_device() > 0 else 'cpu'
    m = torch.tensor(AUDIO_MEAN)
    s = torch.tensor(AUDIO_STD)
    return ((tensor.clone().cpu() - m) / s).to(device)


def deemphasize(wav, coeff=0.97):
    """Undo Kaldi's pre-emphasis filter, y[n] = x[n] - coeff * x[n-1].

    imagebind.data.waveform2melspec calls torchaudio.compliance.kaldi.fbank, which
    pre-emphasizes the waveform by default, so this has to be inverted on playback.
    """
    return scipy.signal.lfilter([1.0], [1.0, -coeff], np.asarray(wav, dtype=np.float64))


def _count_padding_frames(clip, pad_value, exact=False):
    """Count trailing mel frames that ImageBind zero-padded (columns flat at pad_value).

    `clip` is (mel_bins, frames) in the *normalized* domain, where ImageBind's padding is
    `pad_value`. With `exact` the unperturbed model input is expected and the columns are
    compared directly; otherwise (adversarial input) a looser flatness test is used, since the
    attack also perturbs the padded frames.
    """
    for n in range(clip.shape[-1] - 1):
        col = clip[:, -1 - n]
        if exact:
            is_pad = np.allclose(col, pad_value, atol=1e-5)
        else:
            is_pad = col.max() - col.min() < 0.5 and abs(col.mean() - pad_value) < 0.5
        if not is_pad:
            return n
    return max(clip.shape[-1] - 1, 0)


def imagebind_melspec_to_waveform(X, sample_rate=16000, num_mel_bins=128, clip_duration=2,
                                  frame_length_ms=25, frame_shift_ms=10, f_min=20.0,
                                  n_iter=64, trim_padding=True, reference=None):
    """Approximately invert ImageBind's normalized log-mel fbank input back to a waveform.

    X: (clips, 1, mel_bins, frames) or (1, clips, 1, mel_bins, frames), as produced by
    imagebind.data.load_and_transform_audio_data. Each clip is un-normalized, un-logged and
    inverted with `librosa.feature.inverse.mel_to_audio` (the mel filter bank is rebuilt with
    Kaldi's HTK mel scale and no area normalization). The clips are concatenated in the same
    order ImageBind sees them, so the result is 3 * clip_duration long and for sources shorter
    than 6s (whose ImageBind clips overlap) longer than the source file. Phase is estimated by
    Griffin-Lim and Kaldi's per-frame mean removal/pre-emphasis are only approximately undone
    (`deemphasize`), so this is lossy.

    Kaldi's zero-padded frames carry no waveform information, so they are dropped by default
    (ImageBind pads to a 204-frame target_length; a full clip only has 198 real frames). Pass
    the unperturbed model input as `reference` to locate that padding exactly, since an
    adversarial X has its padded frames perturbed too.
    """
    X = X.detach().cpu().float()
    pad_value = (0.0 - AUDIO_MEAN) / AUDIO_STD  # value of the frames Kaldi padded with log-mel 0
    if reference is not None:
        reference = reference.detach().cpu().float()
        if reference.shape != X.shape:
            raise ValueError(f'reference {tuple(reference.shape)} must match X {tuple(X.shape)}')

    n_frames_in = X.shape[-1]
    X_norm = X.reshape(-1, num_mel_bins, n_frames_in).numpy()  # normalized domain, for padding
    X = unnorm_audio(X).numpy().reshape(-1, num_mel_bins, n_frames_in)  # undo Normalize(mean, std)
    if reference is not None:
        reference = reference.reshape(-1, num_mel_bins, n_frames_in).numpy()

    frame_length, frame_shift = int(frame_length_ms / 1000 * sample_rate), \
                                int(frame_shift_ms / 1000 * sample_rate)
    n_fft = 1 << (frame_length - 1).bit_length()  # Kaldi pads frames up to a power of 2
    max_frames = 1 + (clip_duration * sample_rate - frame_length) // frame_shift

    clips = []
    for i, clip in enumerate(X):
        n_frames = min(max_frames, clip.shape[1])
        if trim_padding:
            pad_source = reference[i] if reference is not None else X_norm[i]
            n_frames = max(1, n_frames - _count_padding_frames(
                pad_source[:, :n_frames], pad_value, exact=reference is not None))
        mel_power = np.exp(clip[:, :n_frames])  # undo the log of Kaldi's fbank
        # Output length is inferred from the number of frames (hop * (frames - 1) samples)
        wav = librosa.feature.inverse.mel_to_audio(
            mel_power, sr=sample_rate, n_fft=n_fft, hop_length=frame_shift,
            win_length=frame_length, window='hann', center=True, power=2.0, n_iter=n_iter,
            fmin=f_min, fmax=sample_rate / 2, htk=True, norm=None)
        clips.append(deemphasize(wav))
    return torch.tensor(np.concatenate(clips))


def save_audio(X, path, model_flag, sample_rate=None, reference=None):
    """Save an (optionally adversarial) audio model input as a .wav file.

    imagebind: X is a normalized log-mel spectrogram, so it is inverted to 16kHz audio.
    audioclip: X is already a raw waveform, so it is saved as-is at 44.1kHz.

    `reference` (imagebind only) is the unperturbed input X, used to locate ImageBind's
    zero-padded mel frames; see `imagebind_melspec_to_waveform`.
    """
    import torchaudio

    if model_flag == 'imagebind':
        sample_rate = sample_rate or 16000
        wav = imagebind_melspec_to_waveform(X, sample_rate=sample_rate, reference=reference)
    elif model_flag in ('audioclip', 'audioclip_partial'):
        sample_rate = sample_rate or 44100
        wav = X.detach().cpu().float().reshape(-1)
    else:
        raise NotImplementedError(f'Audio saving is not supported for model {model_flag}')

    wav = wav.to(torch.float32)
    peak = float(wav.abs().max())
    if peak > 1.0:  # avoid clipping when writing
        wav = wav / peak
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), wav[None], sample_rate)  # torchaudio expects (channels, time)
    return wav, sample_rate


def threshold(X, eps, modality, device):
    if modality == 'vision':
        X_unnorm = unnorm(X.data)
        X_max, X_min = norm(torch.clamp(X_unnorm+eps, min=0, max=1)), norm(torch.clamp(X_unnorm-eps, min=0, max=1))
    elif modality == 'thermal':
        X_max, X_min = torch.clamp(X+eps, min=0, max=1), torch.clamp(X-eps, min=0, max=1)
    elif modality == 'audio':
        X_max, X_min = X + eps, X - eps
    return X_max.to(device), X_min.to(device)

def extract_args(exp_name):
    fnm = f'configs/{exp_name}.toml'
    print(f'Loading config from {fnm}...')

    cfg_dict = toml.load(fnm)['general']

    Path(cfg_dict['output_dir']).mkdir(parents=True, exist_ok=True)
    if 'model_flag' in cfg_dict:
        cfg_dict['model_flags'] = [cfg_dict['model_flag']]
        cfg_dict['target_model_flag'] = cfg_dict['model_flag']
    cfg_dict['target_model_flag'] = cfg_dict.get('target_model_flag', None)

    if 'gpu_num' in cfg_dict:
        cfg_dict['gpu_nums'] = [cfg_dict['gpu_num']]

    cfg_dict['jpeg'] = ('jpeg' in cfg_dict) and cfg_dict['jpeg']
    assert (not cfg_dict['jpeg']) or (cfg_dict['modality'] == 'vision')

    if cfg_dict['modality'] == 'vision':
        cfg_dict['epsilon'] = cfg_dict['epsilon'] / 255

    if type(cfg_dict['epochs']) == list:
        cfg_dict['max_epochs'] = max(cfg_dict['epochs'])
    else:
        cfg_dict['max_epochs'] = cfg_dict['epochs']
        cfg_dict['epochs'] = [cfg_dict['epochs']]

    assert cfg_dict['number_images'] % cfg_dict['batch_size'] == 0
    return Namespace(**cfg_dict)

def extract_eval_args(exp_name):
    fnm = f'configs/{exp_name}.toml'
    print(f'Loading config from {fnm}...')

    cfg_dict = toml.load(fnm)['general']

    if 'model_flag' in cfg_dict:
        cfg_dict['model_flags'] = [cfg_dict['model_flag']]
        cfg_dict['target_model_flag'] = cfg_dict['model_flag']
    cfg_dict['target_model_flag'] = cfg_dict.get('target_model_flag', None)

    if 'gpu_num' in cfg_dict:
        cfg_dict['gpu_nums'] = [cfg_dict['gpu_num']]

    assert cfg_dict['eval_type'] in ['adversarial', 'organic', 'transfer']
    if cfg_dict['eval_type'] == 'transfer':
        assert 'adv_file' in cfg_dict

    assert cfg_dict['number_images'] % cfg_dict['batch_size'] == 0
    return Namespace(**cfg_dict)

def jpeg(x, height=224, width=224, rounding=diff_round, quality=80):
    img_tensor = unnorm(x).squeeze(0)
    factor = quality_to_factor(quality)
    y, cb, cr = compress_jpeg(img_tensor, rounding=rounding, factor=factor)
    img_tensor = decompress_jpeg(y, cb, cr, height, width, rounding=rounding, factor=factor)
    return norm(img_tensor)

def pgd_step(model, X, Y, X_min, X_max, lr, modality, device):
    embeds = model.forward(X, modality, normalize=False)
    loss = 1 - criterion(embeds, Y, dim=1)
    update = lr * torch.autograd.grad(outputs=loss.mean(), inputs=X)[0].sign()
    X = (X.detach().cpu() - update.detach().cpu()).to(device)
    X = torch.clamp(X, min=X_min, max=X_max).requires_grad_(True)
    return X, embeds, loss.clone().detach().cpu()

def gpu_num_to_device(gpu_num):
    return f'cuda:{gpu_num}' if torch.cuda.is_available() and gpu_num >= 0 else 'cpu'

def load_model_data_and_dataset(dataset_flag, model_flags, gpus, seed):
    devices = [gpu_num_to_device(g) for g in gpus]  
    models = [load_model(f, devices[i % len(devices)]) for i, f in enumerate(model_flags)]
    dataset = create_dataset(dataset_flag, model=models[0], device=devices[0], seed=seed)
    embeddings = []
    for i, f in enumerate(model_flags):
        fnm = EMBEDDING_FNM.format(dataset_flag, f)
        embeddings.append(get_embeddings(fnm, dataset.label_texts, devices[i % len(devices)],
                                         dataset_flag, models[i]))
    model_data = [
        (m, e, devices[i % len(devices)]) for i, (m, e) in enumerate(zip(models, embeddings))
    ]
    return model_data, dataset

def print_results(ranks, losses, model_flag):
    if type(ranks) == list:
        ranks = np.concatenate(ranks)
    if type(losses) == list:
        losses = np.concatenate(losses)

    top1 = f'{(ranks < 1).mean():.2f}'
    top5 = f'{(ranks < 5).mean():.2f}'
    mean = f'{np.mean(losses):.4f}'
    stddev = f'{np.std(losses):.4f}'
    print(f'{model_flag},{top1},{top5},{mean},{stddev}')
