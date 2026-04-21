import os
import torch
import whisper
from whisper.tokenizer import get_tokenizer


MODEL_NAME_MAPPER = {
    'whisper-tiny'  : 'tiny.en',
    'whisper-tiny-multi'  : 'tiny',
    'whisper-base'  : 'base.en',
    'whisper-base-multi'  : 'base',
    'whisper-small' : 'small.en',
    'whisper-small-multi' : 'small',
    'whisper-medium'  : 'medium.en',
    'whisper-medium-multi'  : 'medium',
    'whisper-large'  : 'large',
}


def get_whisper_download_root():
    explicit_cache_dir = os.getenv('WHISPER_CACHE_DIR')
    if explicit_cache_dir:
        return explicit_cache_dir

    xdg_cache_home = os.getenv('XDG_CACHE_HOME')
    if xdg_cache_home:
        return os.path.join(xdg_cache_home, 'whisper')

    hf_home = os.getenv('HF_HOME')
    if hf_home:
        return os.path.join(hf_home, 'whisper')

    return os.path.join(os.getcwd(), '.cache', 'whisper')

class WhisperModel:
    '''
        Wrapper for Whisper ASR Transcription
    '''
    def __init__(self, model_name='whisper-small', device=torch.device('cpu'), task='transcribe', language='en'):
        self.model_name = model_name
        self.download_root = get_whisper_download_root()
        self.model = whisper.load_model(MODEL_NAME_MAPPER[model_name], device=device, download_root=self.download_root)
        self.task = task
        self.language = language.split('_')[0] # source audio language
        self.tokenizer = get_tokenizer(self.model.is_multilingual, num_languages=self.model.num_languages, language=self.language, task=self.task)

    
    def predict(self, audio='', initial_prompt=None, without_timestamps=False):
        '''
            Whisper decoder output here
        '''
        result = self.model.transcribe(audio, language=self.language, task=self.task, initial_prompt=initial_prompt, without_timestamps=without_timestamps)
        segments = []
        for segment in result['segments']:
            segments.append(segment['text'].strip())
        return ' '.join(segments)


class WhisperModelEnsemble:
    '''
        Wrapper for Whisper ASR
        Ensemble
        Ensure all models are either multi-lingual or English only
    '''
    def __init__(self, model_names=['whisper-small'], device=torch.device('cpu'), task='transcribe', language='en'):
        self.download_root = get_whisper_download_root()
        self.models = [
            whisper.load_model(MODEL_NAME_MAPPER[model_name], device=device, download_root=self.download_root)
            for model_name in model_names
        ]
        self.task = task
        self.language = language # source audio language
        self.tokenizer = get_tokenizer(self.models[0].is_multilingual, num_languages=self.models[0].num_languages, language=self.language, task=self.task)



