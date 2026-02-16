import json
import os
import sys
from argparse import ArgumentParser

import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader

import soundfile as sf
import whisper

from load_multi_audio_files import pad_or_trim


# ----------------------------
# Custom Dataset for audio dir
# ----------------------------
class AudioFolderDataset(Dataset):
    def __init__(self, dir_path):
        self.files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
                      if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg'))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return {"path": self.files[idx]}

def collate_fn(batch):
    return [item["path"] for item in batch]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dir", "-d", type=str, required=True, help="Directory containing audio files")
    parser.add_argument("--whisper-model", "-w", type=str, default="base",
                        help="Whisper model name (tiny/base/small/medium/large[.en])")
    parser.add_argument("--target-frequency", "-t", type=int, default=16000, help="Target frequency for saving")
    parser.add_argument("--save-dir", "-s", type=str, default="",
                        help="Directory to save audio files and transcriptions")
    parser.add_argument("--batch-size", "-b", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--limit", "-n", type=int, default=None,
                        help="Limit number of files to process (for debugging)")
    args = parser.parse_args()

    dir_name_only = os.path.basename(args.dir.rstrip("/"))

    if args.save_dir and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        save_dir = args.save_dir
    else:
        save_dir = args.dir

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load Whisper
    # print(f"Loading Whisper model {args.whisper_model}...", file=sys.stderr)
    model = whisper.load_model(args.whisper_model).to(DEVICE)
    # print("Model loaded.", file=sys.stderr)

    dataset = AudioFolderDataset(args.dir)
    if args.limit is not None:
        dataset.files = dataset.files[:args.limit]

    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True,
                            prefetch_factor=4, collate_fn=collate_fn)

    transcriptions = {}
    model.eval()
    with torch.no_grad():
        for paths_batch in dataloader:
            try:
                # Load audio and convert to torch tensor

                results = [model.transcribe(path) for path in paths_batch]

                for path, res in zip(paths_batch, results):
                    if path not in transcriptions:
                        transcriptions[os.path.basename(path)] = res["text"]

                del results
                torch.cuda.empty_cache()

                # print(f"Transcribed: {os.path.basename(path)}", file=sys.stderr)

            except Exception as e:
                print(f"Failed to process {paths_batch}: {e}", file=sys.stderr)



    # Save all transcriptions to JSON
    json_path = os.path.join(save_dir, f"transcriptions_{args.whisper_model}.json")
    with open(json_path, "w") as f:
        json.dump(transcriptions, f, indent=2)

    print(f"Done transcribing audio files in {dir_name_only}.", file=sys.stderr)
