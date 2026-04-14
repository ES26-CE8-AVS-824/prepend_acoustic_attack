import io
import json
import os
import sys
import warnings
from argparse import ArgumentParser

import datasets as ds
import soundfile as sf
import whisper
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
import torch
from load_multi_audio_files import GPUReadyAudioDataset, collate_audio_pinned, VoiceBankAudioDataset, BadayvedatVCTKAudioDataset
from src.data.vctk import is_vctk_dataset, load_vctk_hf_split

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


def make_save_path(path, directory, ext=".wav"):
    filename = os.path.basename(path)
    if not os.path.splitext(filename)[1]:
        filename += ext
    full_path = os.path.abspath(os.path.join(directory, filename))
    return full_path, filename


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, required=True, help="Dataset on HuggingFace")
    parser.add_argument("--noisy", action="store_true", help="Whether to use noisy audio from VoiceBank+DEMAND")
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

    if not args.dataset.endswith("VoiceBank-DEMAND-16k") and args.noisy:
            print("--noisy flag can only apply to VoiceBank-DEMAND-16k. Are you sure you wanted to specify it?", file=sys.stderr)
            sys.exit(1)
    clean_noisy_file_suffix = "_noisy" if args.noisy else "_clean"

    # -------------------------
    # Load heavy model once
    # -------------------------
    print("Loading attack...", file=sys.stderr)

    SPLIT = "test"
    split_str = f"{SPLIT}[:{args.limit}]" if args.limit is not None else SPLIT
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    options = whisper.DecodingOptions(language="en", task="transcribe")

    muted_audio = np.load(f"audio_attack_segments/{args.whisper_model}.np.npy")
    audio_attack_segment = torch.from_numpy(muted_audio).to(DEVICE)
    model = whisper.load_model(args.whisper_model).to(DEVICE)

    # Config name is dataset-specific
    if "fleurs" in args.dataset:
        config = "en_us" if not args.whisper_model.endswith(".en") else "default"
    else:
        config = None  # VoiceBank-DEMAND-16k needs no config

    # Load dataset
    if is_vctk_dataset(args.dataset):
        hf_dataset = load_vctk_hf_split(SPLIT, limit=args.limit)
    else:
        load_kwargs = dict(split=split_str, trust_remote_code=True)
        if config:
            hf_dataset = ds.load_dataset(args.dataset, config, **load_kwargs)
        else:
            hf_dataset = ds.load_dataset(args.dataset, **load_kwargs)

    # Create dataloader from dataset
    if args.dataset.endswith("VoiceBank-DEMAND-16k"):
        dataset = VoiceBankAudioDataset(hf_dataset, source="noisy" if args.noisy else "clean")
    elif is_vctk_dataset(args.dataset):
        dataset = BadayvedatVCTKAudioDataset(hf_dataset)
    else:  # google/fleurs
        dataset = GPUReadyAudioDataset(hf_dataset)

    dataloader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True,
                            prefetch_factor=4, collate_fn=collate_audio_pinned)

    # -------------------------
    # Check for existing original transcriptions
    # -------------------------
    original_transcriptions_path = os.path.join(args.original_save_dir, f"transcriptions{clean_noisy_file_suffix}_w-{args.whisper_model}.json")
    if os.path.exists(original_transcriptions_path):
        print(f"Found existing original transcriptions at {original_transcriptions_path}, skipping clean Whisper pass.", file=sys.stderr)
        with open(original_transcriptions_path, "r") as f:
            transcriptions_original = json.load(f)
        skip_original = True
    else:
        transcriptions_original = {}
        skip_original = False

    print("Attack loaded. \"Attacking\"", args.limit if args.limit else len(dataloader), "samples...", file=sys.stderr)

    transcriptions_muted = {}
    model.eval()
    with torch.no_grad():
        samples_processed = 0
        for audio_batch, paths, _ in tqdm(dataloader):

            samples_processed += len(audio_batch)

            audio_batch = audio_batch.to(DEVICE, non_blocking=True)  # [B, 480k] pinned

            # --- Clean pass: skip if transcriptions already exist ---
            results_og = [None] * len(paths)
            if not skip_original:
                mels = [whisper.log_mel_spectrogram(audio) for audio in audio_batch]  # [80, 3000] each
                mels = torch.stack(mels)  # [B, 80, 3000]
                # Batch decoder/encoder (90% time, fully parallel)
                results_og = model.decode(mels, options)

            # --- Attack pass: always run ---
            audios_muted = [torch.cat((audio_attack_segment, audio), dim=0)[:30 * args.sampling_frequency] for audio in
                            audio_batch]
            mels_muted = [whisper.log_mel_spectrogram(cat) for cat in audios_muted]
            mels_muted = torch.stack(mels_muted)  # [B, 80, 3000]
            results_muted = model.decode(mels_muted, options)

            for index, (path, audio_og, audio_muted, res_muted) in enumerate(zip(paths, audio_batch, audios_muted, results_muted)):
                og_save_path_full, og_save_path_filename = make_save_path(path, args.original_save_dir)
                adv_save_path_full, adv_save_path_filename = make_save_path(path, args.adv_save_dir)

                if not skip_original:
                    transcriptions_original[og_save_path_filename] = results_og[index].text
                    sf.write(og_save_path_full, audio_og.cpu().numpy(), args.target_frequency, format="WAV")

                transcriptions_muted[adv_save_path_filename] = res_muted.text
                sf.write(adv_save_path_full, audio_muted.cpu().numpy(), args.target_frequency, format="WAV")

            del audio_batch, audios_muted, mels_muted, results_muted
            if not skip_original:
                del mels
            del results_og
            torch.cuda.empty_cache()
    # print(text)
    if not skip_original:
        with open(original_transcriptions_path, "w") as f:
            json.dump(transcriptions_original, f, indent=2)
    with open(os.path.join(args.adv_save_dir, f"transcriptions{clean_noisy_file_suffix}_w-{args.whisper_model}.json"), "w") as f:
        json.dump(transcriptions_muted, f, indent=2)

    # print("Done attacking.", file=sys.stderr)
    # send_message({"filename": "STOP"}, b"")
    # print("Sent shut down signal.", file=sys.stderr)
