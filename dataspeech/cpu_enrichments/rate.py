from phonemizer import phonemize
from phonemizer.backend import EspeakBackend

class SharedPhonemizer:
    _backend = None
    
    @classmethod
    def get_backend(cls, language='el'):
        # Create backend if it doesn't exist or if language changed
        if cls._backend is None or cls._backend.language != language:
            cls._backend = EspeakBackend(
                language=language,
                punctuation_marks=';:,.!?¡¿—…"«»""~/。【】、‥،؟""؛',
                preserve_punctuation=True,
                language_switch='remove-flags'
            )
        return cls._backend

def rate_apply(batch, rank=None, audio_column_name="audio", text_column_name="text"):
    # Get shared backend instance
    backend = SharedPhonemizer.get_backend(language='el')
    
    if isinstance(batch[text_column_name], list):
        speaking_rates = []
        phonemes_list = []
        
        if "speech_duration" in batch:
            for text, audio_duration in zip(batch[text_column_name], batch["speech_duration"]):
                # Use backend directly instead of phonemize function
                phonemes = backend.phonemize([text])[0]
                audio_duration = audio_duration if audio_duration != 0 else 0.01
                speaking_rate = len(phonemes) / audio_duration
                speaking_rates.append(speaking_rate)
                phonemes_list.append(phonemes)
        else:
            for text, audio in zip(batch[text_column_name], batch[audio_column_name]):
                phonemes = backend.phonemize([text])[0]
                sample_rate = audio["sampling_rate"]
                audio_length = len(audio["array"].squeeze()) / sample_rate
                speaking_rate = len(phonemes) / audio_length
                speaking_rates.append(speaking_rate)
                phonemes_list.append(phonemes)
                
        batch["speaking_rate"] = speaking_rates
        batch["phonemes"] = phonemes_list
    else:
        phonemes = backend.phonemize([batch[text_column_name]])[0]
        
        if "speech_duration" in batch:
            audio_length = batch["speech_duration"] if batch["speech_duration"] != 0 else 0.01
        else:
            sample_rate = batch[audio_column_name]["sampling_rate"]
            audio_length = len(batch[audio_column_name]["array"].squeeze()) / sample_rate
            
        speaking_rate = len(phonemes) / audio_length
        batch["speaking_rate"] = speaking_rate
        batch["phonemes"] = phonemes
        
    return batch
