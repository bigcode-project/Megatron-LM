import contextlib
import logging
import math
import sys
import time
import traceback
import typing

import torch


logger = logging.getLogger(__name__)


# A global buffer for holding logged tensor stats.
_tensor_log_stats: list | None = None


@contextlib.contextmanager
def run_and_log_exception():
    try:
        yield
    except Exception:
        logger.critical(traceback.format_exc())
        # TODO: This is needed because ngc crops the logs.
        time.sleep(10)
        sys.exit(1)


def reset_tensor_stats_logging(enabled=True):
    global _tensor_log_stats
    _tensor_log_stats = [] if enabled else None


def get_logged_tensor_stats():
    return _tensor_log_stats


def format_number(x, prec=4, exp_threshold=3):
    digits = 0 if x == 0 else math.log10(abs(x))
    if math.isfinite(digits) and -exp_threshold < math.floor(digits) < prec + exp_threshold:
        return f"{x:.{prec}f}"
    else:
        return f"{x:.{prec-1}e}"


def log_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    scale: float = 1.0,
    level: int = 2,
    storage: bool = False,
    log_fn: typing.Callable[[str], typing.Any] | None = logger.info,
):
    if level < 1:
        return
    save_stats = _tensor_log_stats is not None
    shape = tuple(tensor.shape)
    _, dtype = str(tensor.dtype).split("torch.")
    txt = [
        (None, name, 50),
        ("shape", shape, 18),
        ("dtype", dtype, 9),
        ("device", tensor.device, 7),
    ]
    stats = dict(
        name=name,
        shape=list(shape),
        dtype=dtype,
        device=str(tensor.device),
    )
    if level >= 2 and tensor.device.type != "meta":
        v_float = tensor.float()

        stats.update(
            mu=v_float.mean().item(),
            std=v_float.std().item(),
            stride=tensor.stride(),
            min=v_float.min().item(),
            max=v_float.max().item(),
        )
        txt.extend(
            [
                ("mu", format_number(stats["mu"] * scale), 10),
                ("std", format_number(stats["std"] * scale), 10),
                ("stride", stats["stride"], 20),
            ]
        )
        if storage:
            storage = tensor.untyped_storage()
            storage_float = torch.tensor(storage, dtype=tensor.dtype, device=tensor.device).float()
            stats.update(
                storage=str(storage.data_ptr())[-8:],
                storage_size=storage.size(),
                storage_mu=storage_float.mean().item() * scale,
                storage_std=storage_float.std().item() * scale,
            )
            txt.extend(
                [
                    (f"storage", stats["storage"], 8),
                    (f"s size", f"{stats['storage_size']:,d}", 12),
                    (f"s mu", format_number(stats["storage_mu"]), 10),
                    (f"s std", format_number(stats["storage_std"]), 10),
                ]
            )
        if level >= 3:
            target_samples = 2 ** (level - 3)
            step = max(tensor.numel() // target_samples, 1)
            while step > 1 and any(step % s == 0 and s > 1 for s in shape):
                step -= 1
            samples = tensor.flatten()[: target_samples * step : step].cpu()
            stats.update(samples=samples, step=step)
            samples = [format_number(x) for x in samples.tolist()]
            samples = ",".join(f"{sample:10s}" for sample in samples)
            txt.append((f"{f'samples (step={step})':21s}", f" ({samples})", target_samples * 11 + 3))
    out, len_ = "", 0
    if save_stats:
        _tensor_log_stats.append(stats)
    for prefix, val, col_len in txt:
        prefix = "" if prefix is None else f" {prefix}="
        len_ += col_len + len(prefix) + 1
        out = f"{f'{out}{prefix}{str(val)}':{len_}s}"
    return log_fn(out)


def log_generator(
    name,
    generator: torch.Tensor | torch.Generator | None = None,
    log_fn: typing.Callable[[str], typing.Any] | None = logger.info,
):
    if generator is None:
        generator = torch.cuda.default_generators[torch.cuda.current_device()]
    tensor = generator.get_state() if isinstance(generator, torch.Generator) else generator
    return log_fn(f"{name} {tensor.view(dtype=torch.int64)[-8:].tolist()}")
