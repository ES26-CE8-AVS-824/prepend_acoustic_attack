"""
prepare_vctk_transcripts.py

Extracts ground-truth transcripts from badayvedat/VCTK and writes them
to a JSON file in the same format produced by apply_attack.py:

    { "<path>.wav": "<transcript>", ... }

The path keys match what BadayvedatVCTKAudioDataset produces (speaker_id + __key__),
so they align with the filenames written by apply_attack.py.

Usage:
    python prepare_vctk_transcripts.py --split validation --output-dir ./vctk_val_original
    python prepare_vctk_transcripts.py --split train      --output-dir ./vctk_train_original
    python prepare_vctk_transcripts.py --split validation --output-dir ./vctk_val_original --limit 200
"""

import json
import os
import sys
from argparse import ArgumentParser

import datasets as ds
from torch.utils.data import DataLoader
from tqdm import tqdm

from load_multi_audio_files import BadayvedatVCTKAudioDataset, collate_audio_pinned


DATASET_ID = "badayvedat/VCTK"
DATASET_CONFIG = "default"


def make_save_filename(path: str) -> str:
    """Mirror the make_save_path() logic from apply_attack.py:
    take the basename and ensure it has a .wav extension."""
    filename = os.path.basename(path)
    if not os.path.splitext(filename)[1]:
        filename += ".wav"
    return filename


def main():
    parser = ArgumentParser(description="Extract badayvedat/VCTK ground-truth transcripts to JSON.")
    parser.add_argument(
        "--split", "-p",
        type=str,
        default="validation",
        choices=["train", "validation"],
        help="Which split to use (default: validation).",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        required=True,
        help="Directory where the transcript JSON will be written.",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Number of samples to process (useful for debugging).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    split_str = f"{args.split}[:{args.limit}]" if args.limit is not None else args.split

    print(f"Loading {DATASET_ID} ({DATASET_CONFIG}), split='{split_str}' ...", file=sys.stderr)
    hf_dataset = ds.load_dataset(DATASET_ID, DATASET_CONFIG, split=split_str)

    dataset = BadayvedatVCTKAudioDataset(hf_dataset)

    # Reuse the same DataLoader setup as apply_attack.py for consistency.
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4,
        collate_fn=collate_audio_pinned,
    )

    print(f"Extracting transcripts for {len(dataset)} samples ...", file=sys.stderr)

    transcripts: dict[str, str] = {}
    for _audio_batch, paths, texts in tqdm(dataloader):
        for path, text in zip(paths, texts):
            filename = make_save_filename(path)
            transcripts[filename] = text

    out_path = os.path.join(args.output_dir, "transcriptions_groundtruth.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(transcripts, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(transcripts)} transcripts → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()