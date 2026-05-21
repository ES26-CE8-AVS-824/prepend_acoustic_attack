import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import random
import os
from tqdm import tqdm

from .learn_attack import AudioAttack
from src.tools.tools import set_seeds, AverageMeter, load_audio_tensor



class AudioAttackHallucinate(AudioAttack):
    '''
       Prepend adversarial attack in audio space -- designed to make Whisper hallucinate by minimizing eot token prediction
    '''
    def __init__(self, attack_args, whisper_model, device, lr=1e-3, multiple_model_attack=False, attack_init='random'):
        AudioAttack.__init__(self, attack_args, whisper_model, device, lr=lr, multiple_model_attack=multiple_model_attack, attack_init=attack_init)
        self.max_length = 400


    def _loss(self, logits, seq_len):
        '''
        The (average) log probability of the end of transcript token

        logits: Torch.tensor [batch x seq_len x vocab_size]
        seq_len: Torch.tensor [batch]
        '''
        tgt_id = self._get_tgt_tkn_id()

        # Compute log probabilities over the vocabulary dimension
        sf = nn.Softmax(dim=2)
        log_probs = torch.log(sf(logits))
        
        # Gather the log probabilities for the target positions and target token
        batch_indices = torch.arange(logits.size(0), device=logits.device)
        tgt_probs = log_probs[batch_indices, seq_len-1, tgt_id]


        return -1/torch.mean(tgt_probs)
    

    def train_step(self, train_loader, epoch, print_freq=25):
        '''
            Run one train epoch - Projected Gradient Descent
        '''
        losses = AverageMeter()

        # switch to train mode
        self.audio_attack_model.train()

        for i, (audio, decoder_input, seq_len) in enumerate(train_loader):
            audio = audio.to(self.device)
            audio_size = audio.size(0)
            decoder_input = decoder_input.to(self.device)
            seq_len = seq_len.to(self.device)

            # Forward pass
            logits = self.audio_attack_model(audio, self.whisper_model, decoder_input=decoder_input)
            del audio, decoder_input
            loss = self._loss(logits, seq_len + self.audio_attack_model.len_sot_ids)
            del logits, seq_len

            # Backward pass and update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.attack_args.clip_val != -1:
                max_val = self.attack_args.clip_val
            else:
                max_val = 100000
            with torch.no_grad():  
                self.audio_attack_model.audio_attack_segment.clamp_(min=-1*max_val, max=max_val)
        
            # record loss
            losses.update(loss.item(), audio_size)
            del loss
            if i % print_freq == 0:
                print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\tLoss {losses.val:.5f} ({losses.avg:.5f})')        


    def _pad_sequence(self, tensors, padding_value=0):
        max_length = max(len(tensor) for tensor in tensors)
        padded_tensors = []
        for tensor in tensors:
            padded_tensor = torch.nn.functional.pad(tensor, (0, max_length - len(tensor)), value=padding_value)
            padded_tensors.append(padded_tensor)
        return padded_tensors

    def _prep_dl(self, data, cache_dir, bs=4, shuffle=False):
        '''
        Create batches of audio vectors, token IDs, and text lengths
        '''

        print('Loading and batching audio files and ref token IDs')
        audio_vectors = AudioAttack._load_audio_vectors_cached(data, cache_dir)
        texts = []

        print('audio loading')
        for d in tqdm(data):
            texts.append(d['ref'])

        # Ensure audio vectors are a single tensor for TensorDataset
        if isinstance(audio_vectors, list):
            audio_vectors = torch.stack(audio_vectors, dim=0)

        # Tokenize texts manually, ensuring padding and truncation
        tokenized_texts = []
        text_lengths = []
        print('text tokenization')
        for text in tqdm(texts):
            if self.whisper_model.model_name == 'canary':
                token_ids = self.whisper_model.tokenizer.text_to_ids(text, 'en')[:self.max_length] # assuming reference text is English
            else:
                token_ids = self.whisper_model.tokenizer.encode(text)[:self.max_length]  
            text_lengths.append(len(token_ids))  # Original length before padding
            if len(token_ids) < self.max_length:
                token_ids.extend([0] * (self.max_length - len(token_ids)))  # Pad
            tokenized_texts.append(torch.tensor(token_ids))

        text_token_ids = torch.stack(tokenized_texts, dim=0)
        text_lengths = torch.tensor(text_lengths)

        ds = TensorDataset(audio_vectors, text_token_ids, text_lengths)
        dl = DataLoader(ds, batch_size=bs, shuffle=shuffle)

        return dl

    def train_process(self, train_data, attack_base_path, cache_dir):
        os.makedirs(attack_base_path, exist_ok=True)
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)

        fpath = f'{attack_base_path}/prepend_attack_models'
        os.makedirs(fpath, exist_ok=True)

        train_dl = self._prep_dl(data=train_data, cache_dir=cache_dir, bs=self.attack_args.bs, shuffle=True)

        for epoch in range(self.attack_args.max_epochs):
            print('current lr {:.5e}'.format(self.optimizer.param_groups[0]['lr']))
            self.train_step(train_dl, epoch)

            if epoch==self.attack_args.max_epochs-1 or (epoch+1)%self.attack_args.save_freq==0:
                os.makedirs(f'{fpath}/epoch{epoch+1}', exist_ok=True)
                state = self.audio_attack_model.state_dict()
                torch.save(state, f'{fpath}/epoch{epoch+1}/model.th')
