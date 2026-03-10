import io
import json
import os
import struct
import sys
from argparse import ArgumentParser
from tqdm import tqdm

import datasets as ds
import numpy as np
import soundfile as sf
import torch
import whisper
from torch.utils.data import DataLoader

from load_multi_audio_files import GPUReadyAudioDataset, collate_audio_pinned, VoiceBankAudioDataset

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


def make_save_path(path, directory, ext=".wav"):
    filename = os.path.basename(path)
    if not os.path.splitext(filename)[1]:
        filename += ext
    return os.path.join(os.path.abspath(directory), filename)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, required=True, help="Dataset on HuggingFace")
    parser.add_argument("--whisper-model", "-w", type=str, default="base",
                        help="Whisper model name (tiny/base/small/medium/large[.en])")
    parser.add_argument("--sampling-frequency", "-f", type=int, default=16000, help="Sampling frequency for processing")
    parser.add_argument("--target_frequency", "-t", type=int, default=16000, help="Target frequency for resampling")
    parser.add_argument("--original-save-dir", "-o", type=str, required=True,
                        help="Directory to save original audio segments")
    parser.add_argument("--adv-save-dir", "-s", type=str, required=True,
                        help="Directory to save attacked audio segments")
    parser.add_argument("--limit", "-n", type=int, default=None, help="Number of samples to process (for debugging)")
    args = parser.parse_args()

    if args.original_save_dir and not os.path.exists(args.original_save_dir):
        os.makedirs(args.original_save_dir)
    if args.adv_save_dir and not os.path.exists(args.adv_save_dir):
        os.makedirs(args.adv_save_dir)

    # -------------------------
    # Load heavy model once
    # -------------------------
    print("Loading attack...", file=sys.stderr)

    LANGUAGE = 'en_us' if not args.whisper_model.endswith('.en') else 'default'
    SPLIT = 'test'
    split_str = f"{SPLIT}[:{args.limit}]" if args.limit is not None else SPLIT
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    options = whisper.DecodingOptions(language="en", task="transcribe")

    muted_audio = np.load(f'audio_attack_segments/{args.whisper_model}.np.npy')
    audio_attack_segment = torch.from_numpy(muted_audio).to(DEVICE)
    model = whisper.load_model(args.whisper_model).to(DEVICE)

    hf_dataset = ds.load_dataset(args.dataset, LANGUAGE, split=split_str, trust_remote_code=True)

    if args.dataset.endswith('VoiceBank-DEMAND-16k'):
        dataset = VoiceBankAudioDataset(hf_dataset, source='clean')
    else:  # google/fleurs
        dataset = GPUReadyAudioDataset(hf_dataset)

    dataloader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True,
                            prefetch_factor=4, collate_fn=collate_audio_pinned)

    print("Attack loaded. \"Attacking\"", args.limit if args.limit else len(dataloader), "samples...", file=sys.stderr)

    transcriptions_original = {}
    transcriptions_muted = {}
    model.eval()
    with torch.no_grad():
        samples_processed = 0
        for audio_batch, paths in tqdm(dataloader):

            samples_processed += len(audio_batch)

            audio_batch = audio_batch.to(DEVICE, non_blocking=True)  # [B, 480k] pinned

            mels = [whisper.log_mel_spectrogram(audio) for audio in audio_batch]  # [80, 3000] each
            mels = torch.stack(mels)  # [B, 80, 3000]
            # Batch decoder/encoder (90% time, fully parallel)
            results_og = model.decode(mels, options)

            audios_muted = [torch.cat((audio_attack_segment, audio), dim=0)[:30 * args.sampling_frequency] for audio in
                            audio_batch]
            mels_muted = [whisper.log_mel_spectrogram(cat) for cat in audios_muted]
            mels_muted = torch.stack(mels_muted)  # [B, 80, 3000]
            results_muted = model.decode(mels_muted, options)

            buffer = io.BytesIO()
            for path, audio_og, res_og, audio_muted, res_muted in zip(paths, audio_batch, results_og, audio_muted,
                                                                      results_muted):
                og_save_path = make_save_path(path, args.original_save_dir)
                adv_save_path = make_save_path(path, args.adv_save_dir)
                transcriptions_original[og_save_path] = res_og.text
                transcriptions_muted[save_path] = res_muted.text
                sf.write(og_save_path, audio.cpu().numpy(), args.target_frequency, format="WAV")
                sf.write(adv_save_path, audio_muted.cpu().numpy(), args.target_frequency, format="WAV")

            del audio_batch, mels, results, audios_muted, mels_muted, results_muted
            torch.cuda.empty_cache()
    # print(text)
    with open(os.path.join(args.original_save_dir, f"transcriptions_{args.whisper_model}.json"), "w") as f:
        json.dump(transcriptions_original, f, indent=2)
    with open(os.path.join(args.adv_save_dir, f"transcriptions_{args.whisper_model}.json"), "w") as f:
        json.dump(transcriptions_muted, f, indent=2)

    # print("Done attacking.", file=sys.stderr)
    # send_message({"filename": "STOP"}, b"")
    # print("Sent shut down signal.", file=sys.stderr)
