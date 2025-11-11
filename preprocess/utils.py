from pathlib import Path
import os
import argparse
from tqdm import tqdm
import pandas as pd
import torch
import esm


def parse_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--root", type=Path)
    return parser.parse_args()


def select_undone_items(input_list: list[tuple[str, str]], output_dir: Path) -> list[tuple[str, str]]:
    all_items = [i[0] for i in input_list]  # Protein names
    done_items = [i.split('.')[0] for i in os.listdir(output_dir) if i.endswith('.pt')]
    undone_items = list(set(all_items).difference(set(done_items)))  # undone PDB names
    return [pair for pair in input_list if pair[0] in undone_items]
