import io
import json
import os
import struct
import sys
from argparse import ArgumentParser

import datasets as ds
import numpy as np
import soundfile as sf
import torch
import whisper
from torch.utils.data import DataLoader

from load_multi_audio_files import GPUReadyAudioDataset, collate_audio_pinned

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


# -----------------------------------
# Utility: read/write framed messages
# -----------------------------------
def read_exact(n):
    data = b''
    while len(data) < n:
        chunk = sys.stdin.buffer.read(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def receive_message():
    # Read header
    header_len_bytes = read_exact(4)
    if header_len_bytes is None:
        return None

    header_len = struct.unpack(">I", header_len_bytes)[0]
    header_json = read_exact(header_len)
    header = json.loads(header_json.decode())

    # Read audio
    audio_len = struct.unpack(">I", read_exact(4))[0]
    audio_bytes = read_exact(audio_len)

    return header, audio_bytes


def send_message(header, audio_bytes):
    header_bytes = json.dumps(header).encode()
    sys.stdout.buffer.write(struct.pack(">I", len(header_bytes)))
    sys.stdout.buffer.write(header_bytes)
    sys.stdout.buffer.write(struct.pack(">I", len(audio_bytes)))
    sys.stdout.buffer.write(audio_bytes)
    sys.stdout.buffer.flush()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, required=True, help="Dataset on HuggingFace")
    parser.add_argument("--whisper-model", "-w", type=str, default="base",
                        help="Whisper model name (tiny/base/small/medium/large[.en])")
    parser.add_argument("--sampling-frequency", "-f", type=int, default=16000, help="Sampling frequency for processing")
    parser.add_argument("--target_frequency", "-t", type=int, default=16000, help="Target frequency for resampling")
    parser.add_argument("--original-save-dir", "-o", type=str, default="",
                        help="Directory to save original audio segments")
    parser.add_argument("--save-dir", "-s", type=str, default="", help="Directory to save attacked audio segments")
    parser.add_argument("--limit", "-n", type=int, default=None, help="Number of samples to process (for debugging)")
    args = parser.parse_args()

    if args.original_save_dir and not os.path.exists(args.original_save_dir):
        os.makedirs(args.original_save_dir)
    if args.save_dir and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # -------------------------
    # Load heavy model once
    # -------------------------
    print("Loading attack...", file=sys.stderr)

    LANGUAGE = 'default' if args.whisper_model.endswith('.en') else 'en_us'
    SPLIT = 'test'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    options = dict(language="en", task="transcribe")

    muted_audio = np.load(f'audio_attack_segments/{args.whisper_model}.np.npy')
    audio_attack_segment = torch.from_numpy(muted_audio).to(DEVICE)
    model = whisper.load_model(args.whisper_model).to(DEVICE)

    dataset = GPUReadyAudioDataset(
        ds.load_dataset(args.dataset, LANGUAGE, split=SPLIT, trust_remote_code=True))
    dataloader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True,
                            prefetch_factor=4, collate_fn=collate_audio_pinned)

    if args.limit is not None:
        # Calculate number of batches needed
        num_batches = (args.limit + dataloader.batch_size - 1) // dataloader.batch_size
        dataloader = list(dataloader)[:num_batches]

    print("Attack loaded. \"Attacking\"", args.limit if args.limit else len(dataloader), "samples...", file=sys.stderr)

    transcriptions_original = {}
    transcriptions_muted = {}
    model.eval()
    with torch.no_grad():
        samples_processed = 0
        for audio_batch, paths in dataloader:
            if args.limit is not None and samples_processed >= args.limit:
                break

            # Only process up to the limit
            if args.limit is not None:
                remaining = args.limit - samples_processed
                if remaining < len(audio_batch):
                    audio_batch = audio_batch[:remaining]
                    paths = paths[:remaining]

            samples_processed += len(audio_batch)

            audio_batch = audio_batch.to(DEVICE, non_blocking=True)  # [B, 480k] pinned

            mels = [whisper.log_mel_spectrogram(audio) for audio in audio_batch]  # [80, 3000] each
            mels = torch.stack(mels)  # [B, 80, 3000]
            # Batch decoder/encoder (90% time, fully parallel)
            options = whisper.DecodingOptions()
            results = model.decode(mels, options)

            for path, res in zip(paths, results):
                transcriptions_original[os.path.basename(path)] = res.text

            audios_muted = [torch.cat((audio_attack_segment, audio), dim=0)[:30 * args.sampling_frequency] for audio in
                            audio_batch]
            mels_muted = [whisper.log_mel_spectrogram(cat) for cat in audios_muted]
            mels_muted = torch.stack(mels_muted)  # [B, 80, 3000]
            results_muted = model.decode(mels_muted, options)
            for path, res in zip(paths, results_muted):
                transcriptions_muted[os.path.basename(path)] = res.text
            buffer = io.BytesIO()
            if args.original_save_dir:
                for path, audio in zip(paths, audio_batch):
                    og_save_path = os.path.join(args.original_save_dir, os.path.basename(path))
                    sf.write(og_save_path, audio.cpu().numpy(), args.target_frequency, format="WAV")
            if args.save_dir:
                for path, audio_muted in zip(paths, audios_muted):
                    adv_save_path = os.path.join(args.save_dir, os.path.basename(path))
                    sf.write(adv_save_path, audio_muted.cpu().numpy(), args.target_frequency, format="WAV")
            # for audio_muted in audios_muted:
            # sf.write(buffer, audio_muted.cpu().numpy(), args.target_frequency, format="WAV")
            # out_bytes = buffer.getvalue()
            # response_header = {
            #    "filename": os.path.basename(path),
            #    "original_transcription": text[path][0],
            #    "muted_transcription": text[path][1],
            # }
            # send_message(response_header, out_bytes)
            # for path in paths:
            #    print("Attacked", os.path.basename(path), file=sys.stderr)

            del audio_batch, mels, results, audios_muted, mels_muted, results_muted
            torch.cuda.empty_cache()
    # print(text)
    with open(os.path.join(args.original_save_dir, f"transcriptions_{args.whisper_model}.json"), "w") as f:
        json.dump(transcriptions_original, f, indent=2)
    with open(os.path.join(args.save_dir, f"transcriptions_{args.whisper_model}.json"), "w") as f:
        json.dump(transcriptions_muted, f, indent=2)

    # print("Done attacking.", file=sys.stderr)
    # send_message({"filename": "STOP"}, b"")
    # print("Sent shut down signal.", file=sys.stderr)
