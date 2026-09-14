"""Deterministic execution and RNG checkpoint utilities."""
from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return

    if "python" in state:
        random.setstate(state["python"])

    if "numpy" in state:
        np_state = state["numpy"]
        if isinstance(np_state, list):
            np_state = tuple(np_state)
        if isinstance(np_state, tuple) and len(np_state) >= 2:
            values = list(np_state)
            if not isinstance(values[1], np.ndarray):
                values[1] = np.asarray(values[1], dtype=np.uint32)
            np_state = tuple(values)
        np.random.set_state(np_state)

    if "torch" in state and state["torch"] is not None:
        torch_state = state["torch"]
        if not torch.is_tensor(torch_state):
            torch_state = torch.tensor(torch_state, dtype=torch.uint8, device="cpu")
        else:
            torch_state = torch_state.detach().to(device="cpu", dtype=torch.uint8)
        torch.set_rng_state(torch_state)

    if torch.cuda.is_available() and state.get("cuda") is not None:
        cuda_states = []
        for cuda_state in state["cuda"]:
            if not torch.is_tensor(cuda_state):
                cuda_state = torch.tensor(cuda_state, dtype=torch.uint8, device="cpu")
            else:
                cuda_state = cuda_state.detach().to(device="cpu", dtype=torch.uint8)
            cuda_states.append(cuda_state)

        current_devices = torch.cuda.device_count()
        if len(cuda_states) == current_devices:
            torch.cuda.set_rng_state_all(cuda_states)
        elif current_devices > 0 and cuda_states:
            for device_idx, cuda_state in enumerate(cuda_states[:current_devices]):
                torch.cuda.set_rng_state(cuda_state, device=device_idx)
