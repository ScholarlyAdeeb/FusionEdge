"""Training progress bar.

Why this exists: a 60-epoch run on 1083 pairs is ~11 minutes, and without a bar
the only signal is a log line every 50 steps. During an experiment sweep that
means repeatedly checking a file to answer "how far along is it?".

Design notes
------------
* **Auto-disable when stderr is not a TTY.** Tests, CI and redirected output get
  no escape sequences and no interleaved junk in the log file; an interactive
  terminal gets the bar. This is the same rule tqdm itself uses, applied by
  default so a run never needs to be configured for it.
* **Forceable.** `progress: always` / `progress: never` in the config overrides
  detection, which matters when someone tails the log through a pager or a
  multiplexer and wants a stable, line-oriented log.
* **Loss in the bar.** The bar shows a running mean rather than the instantaneous
  batch loss, because per-batch loss on 128x128 crops is noisy enough that
  watching it move is misleading.
* **ETA from measured throughput**, not from the config, so it accounts for a
  slow first epoch (cuDNN autotuning, dataloader warmup).
"""

from __future__ import annotations

import sys
import time
from typing import Any, Iterable, Iterator

try:  # tqdm is a declared dependency, but degrade rather than crash if absent
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover - exercised only in a broken install
    _tqdm = None


def bar_enabled(mode: str = "auto") -> bool:
    """Resolve the `progress` setting against whether stderr is interactive."""
    mode = (mode or "auto").lower()
    if mode in ("never", "off", "false", "0"):
        return False
    if mode in ("always", "on", "true", "1"):
        return _tqdm is not None
    return _tqdm is not None and sys.stderr.isatty()


class _NullBar:
    """Minimal stand-in so callers need no `if bar:` branches.

    It mirrors tqdm's interface closely enough that the real bar and this one can
    be swapped freely -- including `set_postfix_str`, which was missed the first
    time and surfaced as an AttributeError only when the bar was disabled, i.e.
    in exactly the environment where nobody was watching.
    """

    def __init__(self) -> None:
        self.n = 0
        self.total = 0
        self.desc = ""

    def update(self, n: int = 1) -> None:
        self.n += n

    def reset(self, total: int | None = None) -> None:
        if total is not None:
            self.total = int(total)
        self.n = 0

    def set_postfix(self, **kwargs: Any) -> None:
        pass

    def set_postfix_str(self, s: str = "", **kwargs: Any) -> None:
        pass

    def set_description(self, desc: str | None = None, refresh: bool = True) -> None:
        if desc is not None:
            self.desc = desc

    def write(self, msg: str) -> None:
        sys.stderr.write(msg + "\n")

    def close(self) -> None:
        pass

    def __enter__(self) -> "_NullBar":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class ProgressBar:
    """A progress bar with a running-mean loss readout, or a no-op stand-in.

    Args:
        total: number of items (optimizer steps, here).
        desc: leading label, usually "epoch 3/60".
        mode: 'auto' | 'always' | 'never'.
        unit: item label.
    """

    def __init__(
        self,
        total: int,
        *,
        desc: str = "",
        mode: str = "auto",
        unit: str = "step",
        leave: bool = True,
    ):
        self.total = max(int(total), 0)
        self.enabled = bar_enabled(mode)
        self._loss_sum = 0.0
        self._loss_n = 0
        self._bar: Any

        if not self.enabled:
            bar = _NullBar()
            bar.total = self.total
            bar.desc = desc
            self._bar = bar
            return

        # dynamic_ncols keeps the bar aligned with the terminal; mininterval
        # throttles redraws so the bar does not dominate the CPU on short epochs.
        self._bar = _tqdm(
            total=self.total,
            desc=desc,
            unit=unit,
            leave=leave,
            dynamic_ncols=True,
            mininterval=0.3,
            bar_format=(
                "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            ),
        )

    # ------------------------------------------------------------------ #
    @property
    def n(self) -> int:
        """Items consumed so far."""
        return self._bar.n

    def update(self, n: int = 1, loss: float | None = None) -> None:
        """Advance the bar, optionally recording a batch loss for the mean."""
        if loss is not None:
            self._loss_sum += float(loss)
            self._loss_n += 1
        self._bar.update(n)

    def set_description(self, desc: str) -> None:
        self._bar.set_description(desc)

    def set_lr(self, lr: float) -> None:
        """Show the current learning rate alongside the loss."""
        self._bar.set_postfix_str(self._postfix(lr))

    def _postfix(self, lr: float | None = None) -> str:
        parts = []
        if self._loss_n:
            parts.append(f"loss {self._loss_sum / self._loss_n:.4f}")
        if lr is not None:
            parts.append(f"lr {lr:.2e}")
        return ", ".join(parts)

    def write(self, msg: str) -> None:
        """Print a line above the bar without corrupting it."""
        if self.enabled:
            self._bar.write(msg)
        else:
            sys.stderr.write(msg + "\n")

    def reset(self, total: int | None = None, desc: str | None = None) -> None:
        if total is not None:
            self.total = int(total)
        self._loss_sum = 0.0
        self._loss_n = 0
        if self.enabled:
            self._bar.reset(total=self.total)
            if desc is not None:
                self._bar.set_description(desc)

    def close(self) -> None:
        self._bar.close()

    def __enter__(self) -> "ProgressBar":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def iter_with_bar(items: Iterable[Any], bar: ProgressBar) -> Iterator[Any]:
    """Iterate `items`, advancing `bar` by one per item."""
    for item in items:
        yield item
        bar.update(1)


class Timer:
    """Wall-clock timer used for epoch durations in the logs."""

    def __init__(self) -> None:
        self.t0 = time.time()

    def reset(self) -> float:
        prev = self.t0
        self.t0 = time.time()
        return time.time() - prev

    @property
    def elapsed(self) -> float:
        return time.time() - self.t0