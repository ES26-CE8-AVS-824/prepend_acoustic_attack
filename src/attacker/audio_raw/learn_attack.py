import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import random
import os
import hashlib
import json
from tqdm import tqdm

from .base import AudioBaseAttacker
from src.tools.tools import AverageMeter, load_audio_tensor



class AudioAttack(AudioBaseAttacker):
    '''
       Prepend adversarial attack in audio space -- designed to mute Whisper by maximizing eot token as first generated token
    '''
    def __init__(self, attack_args, whisper_model, device, lr=1e-3, multiple_model_attack=False, attack_init='random'):
        AudioBaseAttacker.__init__(self, attack_args, whisper_model, device, attack_init=attack_init)
        self.audio_attack_model.multiple_model_attack = multiple_model_attack
        self.optimizer = torch.optim.AdamW(self.audio_attack_model.parameters(), lr=lr, eps=1e-8)


    def _loss(self, logits):
        '''
        The (average) negative log probability of the end of transcript token

        logits: Torch.tensor [batch x vocab_size]
        '''
        tgt_id = self._get_tgt_tkn_id()
        sf = nn.Softmax(dim=1)
        log_probs = torch.log(sf(logits))
        tgt_probs = log_probs[:,tgt_id].squeeze()
        return -1*torch.mean(tgt_probs)
    

    def train_step(self, train_loader, epoch, print_freq=25):
        '''
            Run one train epoch - Projected Gradient Descent
        '''
        losses = AverageMeter()

        # switch to train mode
        self.audio_attack_model.train()

        for i, (audio) in enumerate(train_loader):
            audio = audio[0].to(self.device)
            n = audio.size(0)

            # Forward pass
            logits = self.audio_attack_model(audio, self.whisper_model)[:,-1,:].squeeze(dim=1)
            del audio
            loss = self._loss(logits)
            del logits

            # Backward pass and update
            self.optimizer.zero_grad()
            loss.backward()
            # print(self.audio_attack_model.audio_attack_segment.grad)
            self.optimizer.step()

            if self.attack_args.clip_val != -1:
                max_val = self.attack_args.clip_val
            else:
                max_val = 100000
            with torch.no_grad():  
                self.audio_attack_model.audio_attack_segment.clamp_(min=-1*max_val, max=max_val)
        
            # record loss
            losses.update(loss.item(), n)
            del loss
            if i % print_freq == 0:
                print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\tLoss {losses.val:.5f} ({losses.avg:.5f})')        


    @staticmethod
    def _pad_sequence(tensors, padding_value=0):
        max_length = max(len(tensor) for tensor in tensors)
        padded_tensors = []
        for tensor in tensors:
            padded_tensor = torch.nn.functional.pad(tensor, (0, max_length - len(tensor)), value=padding_value)
            padded_tensors.append(padded_tensor)
        return padded_tensors

    @staticmethod
    def _load_audio_vectors_cached(data, cache_dir):
        if cache_dir is not None:
            cache_path = AudioAttack._get_audio_cache_path(data, cache_dir)
            if os.path.isfile(cache_path):
                print(f'Loading cached audio tensor from {cache_path}')
                return torch.load(cache_path, map_location='cpu')

        print('Loading and batching audio files')
        raw_vectors = []
        for d in tqdm(data):
            raw_vectors.append(load_audio_tensor(d['audio']))

        padded = AudioAttack._pad_sequence(raw_vectors)
        del raw_vectors
        audio_vectors = torch.stack(padded, dim=0)
        del padded

        if cache_dir is not None:
            torch.save(audio_vectors, cache_path)
            print(f'Saved audio tensor cache to {cache_path}')
        return audio_vectors

    @staticmethod
    def _serialize_audio_cache_key(audio):
        if isinstance(audio, str):
            path = os.path.abspath(audio)
            if os.path.isfile(path):
                stat = os.stat(path)
                return {
                    'type': 'path',
                    'path': path,
                    'mtime_ns': stat.st_mtime_ns,
                    'size': stat.st_size,
                }
            return {'type': 'path', 'path': path, 'missing': True}

        if isinstance(audio, dict):
            if audio.get('path'):
                return AudioAttack._serialize_audio_cache_key(audio['path'])

            serializable_audio = {}
            for key, value in audio.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    serializable_audio[key] = value
                elif isinstance(value, bytes):
                    serializable_audio[key] = {
                        'bytes_len': len(value),
                        'sha256': hashlib.sha256(value).hexdigest(),
                    }
                elif torch.is_tensor(value):
                    tensor_bytes = value.detach().cpu().contiguous().numpy().tobytes()
                    serializable_audio[key] = {
                        'shape': tuple(value.shape),
                        'dtype': str(value.dtype),
                        'sha256': hashlib.sha256(tensor_bytes).hexdigest(),
                    }
                elif hasattr(value, 'shape'):
                    array_bytes = value.tobytes()
                    serializable_audio[key] = {
                        'shape': tuple(value.shape),
                        'dtype': str(getattr(value, 'dtype', type(value))),
                        'sha256': hashlib.sha256(array_bytes).hexdigest(),
                    }
                elif isinstance(value, (list, tuple)):
                    sequence_repr = json.dumps(list(value), default=str).encode('utf-8')
                    serializable_audio[key] = {
                        'len': len(value),
                        'sha256': hashlib.sha256(sequence_repr).hexdigest(),
                    }
                else:
                    serializable_audio[key] = str(type(value))
            return {'type': 'dict', 'value': serializable_audio}

        if torch.is_tensor(audio):
            tensor_bytes = audio.detach().cpu().contiguous().numpy().tobytes()
            return {
                'type': 'tensor',
                'shape': tuple(audio.shape),
                'dtype': str(audio.dtype),
                'sha256': hashlib.sha256(tensor_bytes).hexdigest(),
            }

        return {'type': type(audio).__name__, 'repr': str(audio)}

    @staticmethod
    def _get_audio_cache_path(data, cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

        cache_descriptor = [
            AudioAttack._serialize_audio_cache_key(d['audio'])
            for d in data
        ]
        cache_key = hashlib.sha256(
            json.dumps(cache_descriptor, sort_keys=True).encode('utf-8')
        ).hexdigest()
        return os.path.join(cache_dir, f'audio_vectors_{cache_key}.pt')

    @staticmethod
    def _prep_dl(data, cache_dir, bs=16, shuffle=False):
        '''
        Create batch of audio vectors
        '''

        audio_vectors = AudioAttack._load_audio_vectors_cached(data, cache_dir)
        ds = TensorDataset(audio_vectors)
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=16)


    def train_process(self, train_data, attack_base_path, cache_dir):

        os.makedirs(attack_base_path, exist_ok=True)
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)

        fpath = f'{attack_base_path}/prepend_attack_models'
        os.makedirs(fpath, exist_ok=True)

        train_dl = AudioAttack._prep_dl(data=train_data, cache_dir=cache_dir, bs=self.attack_args.bs, shuffle=True)

        for epoch in range(self.attack_args.max_epochs):
            # train for one epoch
            print('current lr {:.5e}'.format(self.optimizer.param_groups[0]['lr']))
            self.train_step(train_dl, epoch)

            if epoch==self.attack_args.max_epochs-1 or (epoch+1)%self.attack_args.save_freq==0:
                # save model at this epoch
                os.makedirs(f'{fpath}/epoch{epoch+1}', exist_ok=True)
                state = self.audio_attack_model.state_dict()
                torch.save(state, f'{fpath}/epoch{epoch+1}/model.th')
