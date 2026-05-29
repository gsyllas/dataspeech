import torch 
import penn


# Here we'll use a 10 millisecond hopsize
hopsize = .01

# Provide a sensible frequency range given your domain and model
fmin = 30.
fmax = 1000.

# Select a checkpoint to use for inference. Selecting None will
# download and use FCNF0++ pretrained on MDB-stem-synth and PTDB
checkpoint = None

# Centers frames at hopsize / 2, 3 * hopsize / 2, 5 * hopsize / 2, ...
center = 'half-hop'

# (Optional) Linearly interpolate unvoiced regions below periodicity threshold
interp_unvoiced_at = .065

# PENN reflect-pads internally by about 472 samples for this configuration.
# Very short clips can be shorter than that after resampling, which makes
# torch's reflect padding fail. Pad them with silence before handing off.
min_penn_audio_samples = 1024


def _prepare_waveform(sample):
    waveform = torch.tensor(sample["array"]).float()
    if waveform.ndim == 0:
        waveform = waveform.reshape(1)
    elif waveform.ndim == 2:
        if waveform.shape[0] <= 2:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.mean(dim=-1)

    waveform = waveform.reshape(1, -1)
    if waveform.shape[-1] < min_penn_audio_samples:
        waveform = torch.nn.functional.pad(
            waveform,
            (0, min_penn_audio_samples - waveform.shape[-1]),
            mode="constant",
            value=0.0,
        )
    return waveform


def pitch_apply(batch, rank=None, audio_column_name="audio", output_column_name="utterance_pitch", penn_batch_size=4096):
    if isinstance(batch[audio_column_name], list):  
        utterance_pitch_mean = []
        utterance_pitch_std = []
        for sample in batch[audio_column_name]:
            # Infer pitch and periodicity
            pitch, periodicity = penn.from_audio(
                _prepare_waveform(sample),
                sample["sampling_rate"],
                hopsize=hopsize,
                fmin=fmin,
                fmax=fmax,
                checkpoint=checkpoint,
                batch_size=penn_batch_size,
                center=center,
                interp_unvoiced_at=interp_unvoiced_at,
                gpu=(rank or 0)% torch.cuda.device_count() if torch.cuda.device_count() > 0 else rank
                )
            
            utterance_pitch_mean.append(pitch.mean().cpu())
            utterance_pitch_std.append(pitch.std().cpu())
            
        batch[f"{output_column_name}_mean"] = utterance_pitch_mean 
        batch[f"{output_column_name}_std"] = utterance_pitch_std 
    else:
        sample = batch[audio_column_name]
        pitch, periodicity = penn.from_audio(
                _prepare_waveform(sample),
                sample["sampling_rate"],
                hopsize=hopsize,
                fmin=fmin,
                fmax=fmax,
                checkpoint=checkpoint,
                batch_size=penn_batch_size,
                center=center,
                interp_unvoiced_at=interp_unvoiced_at,
                gpu=(rank or 0)% torch.cuda.device_count() if torch.cuda.device_count() > 0 else rank
                )        
        batch[f"{output_column_name}_mean"] = pitch.mean().cpu()
        batch[f"{output_column_name}_std"] = pitch.std().cpu()

    return batch
