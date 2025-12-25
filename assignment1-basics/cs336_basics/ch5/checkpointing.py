import torch
import os
from typing import BinaryIO, IO
import pickle

def save_tokenizer_assets(vocab, merges, vocab_path, merges_path):
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    with open(merges_path, 'wb') as f:
        pickle.dump(merges, f)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration
    }
    if isinstance(out, (str, os.PathLike)):
        with open(out, 'wb') as f:
            pickle.dump(state, f)
    else:
        pickle.dump(state, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    if isinstance(src, (str, os.PathLike)):
        with open(src, 'rb') as f:
            state = pickle.load(f)
    else:
        state = pickle.load(src)

    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    return state['iteration']