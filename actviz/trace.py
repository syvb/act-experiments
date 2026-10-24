"""A recorded forward pass: every intermediate tensor, plus a description of how
the viewer should lay those tensors out as pages of equations.

The viewer knows nothing about any particular model. A model adapter (see
`actviz.models`) runs the forward pass in numpy, registers each intermediate with
`Trace.add`, and describes the computation as sections → pages → blocks. The block
and term vocabulary is documented in the README; everything the page shows comes
from here, so the page never has to do model arithmetic itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# How a tensor's values map to color in the viewer.
#   signed   - diverging scale centred on 0 (activations, scores)
#   positive - sequential scale from 0 to a robust max (norms, vocab probabilities)
#   prob     - sequential scale fixed to [0, 1] (attention patterns)
KINDS = ("signed", "positive", "prob")


@dataclass
class TensorRecord:
    id: str
    data: np.ndarray
    axes: list[str]
    kind: str
    label: str
    desc: str
    chunk: str
    scale: float | None = None  # overrides the robust color-scale limit
    percent: bool = False  # show values as percentages in tooltips
    # Set when this tensor is a slice of another one (`of`), taken at `at`.
    of: str | None = None
    at: dict[str, int] = field(default_factory=dict)


class Trace:
    def __init__(self, *, model: dict, prompt: str, tokens: list[dict], axes: dict[str, dict]):
        """`axes` maps an axis name to how its indices are labelled in the viewer:
        {"labels": "tokens"} (the prompt tokens), {"labels": "<name>"} (a list
        registered with `add_labels`), {"labels": [...]} (inline), or
        {"prefix": "H"} (H0, H1, ...)."""
        self.model = model
        self.prompt = prompt
        self.tokens = tokens
        self.axes = axes
        self.tensors: dict[str, TensorRecord] = {}
        self.labels: dict[str, list[str]] = {}
        self.sections: list[Section] = []
        self.extra: dict[str, Any] = {}

    def add(
        self,
        id: str,
        data: np.ndarray,
        axes: list[str],
        *,
        chunk: str,
        kind: str = "signed",
        label: str | None = None,
        desc: str = "",
        scale: float | None = None,
        percent: bool = False,
    ) -> str:
        data = np.asarray(data, dtype=np.float32)
        if id in self.tensors:
            raise ValueError(f"duplicate tensor id {id!r}")
        if data.ndim != len(axes):
            raise ValueError(f"{id}: {data.ndim}-d data but axes {axes}")
        if kind not in KINDS:
            raise ValueError(f"{id}: unknown kind {kind!r}")
        self.tensors[id] = TensorRecord(id, data, list(axes), kind, label or id, desc, chunk, scale, percent)
        return id

    def alias(self, id: str, of: str, at: dict[str, int], *, label: str | None = None, desc: str = "") -> str:
        """Register `id` as the slice `of[at]` without storing its data twice.
        It still gets its own color scale, computed from the slice."""
        base = self.tensors[of]
        index = tuple(at.get(a, slice(None)) for a in base.axes)
        axes = [a for a in base.axes if a not in at]
        rec = TensorRecord(id, base.data[index], axes, base.kind, label or id, desc, base.chunk, None, base.percent, of, dict(at))
        self.tensors[id] = rec
        return id

    def add_labels(self, name: str, labels: list[str]) -> None:
        self.labels[name] = labels

    def section(self, id: str, title: str, short: str | None = None) -> "Section":
        s = Section(id, title, short)
        self.sections.append(s)
        return s

    def validate(self) -> None:
        """Check that every tensor a block refers to exists and that views name real axes."""
        for s in self.sections:
            for p in s.pages:
                for ref in p.tensor_refs():
                    tid, view = ref["tensor"], ref
                    if tid not in self.tensors:
                        raise ValueError(f"page {p.id}: unknown tensor {tid!r}")
                    axes = self.tensors[tid].axes
                    for key in ("rows", "cols"):
                        for a in view.get(key) or []:
                            if a not in axes:
                                raise ValueError(f"page {p.id}: {tid} has no axis {a!r} (axes {axes})")
                    for a in (view.get("fix") or {}):
                        if a not in axes:
                            raise ValueError(f"page {p.id}: {tid} has no axis {a!r} to fix (axes {axes})")


class Section:
    def __init__(self, id: str, title: str, short: str | None = None):
        self.id = id
        self.title = title
        self.short = short
        self.pages: list[Page] = []

    def page(self, id: str, title: str, short: str | None = None) -> "Page":
        p = Page(id, title, short or title)
        self.pages.append(p)
        return p

    def to_json(self) -> dict:
        out = {"id": self.id, "title": self.title, "pages": [p.to_json() for p in self.pages]}
        if self.short:
            out["short"] = self.short
        return out


class Page:
    def __init__(self, id: str, title: str, short: str):
        self.id = id
        self.title = title
        self.short = short
        self.blocks: list[dict] = []

    def text(self, md: str) -> "Page":
        """A paragraph. Supports **bold**, *italic* and `code`; blank lines split paragraphs."""
        self.blocks.append({"type": "text", "text": md.strip()})
        return self

    def equation(self, title: str, *lines: dict, formula: str | None = None, note: str | None = None) -> "Page":
        """A step of the computation; see `equation`."""
        self.blocks.append(equation(title, *lines, formula=formula, note=note))
        return self

    def block(self, type: str, **kw) -> "Page":
        self.blocks.append({"type": type, **kw})
        return self

    def tensor_refs(self):
        def walk(x):
            if isinstance(x, dict):
                if "tensor" in x and isinstance(x["tensor"], str):
                    yield x
                for v in x.values():
                    yield from walk(v)
            elif isinstance(x, list):
                for v in x:
                    yield from walk(v)

        yield from walk(self.blocks)

    def to_json(self) -> dict:
        return {"id": self.id, "title": self.title, "short": self.short, "blocks": self.blocks}


# ---- equation terms -------------------------------------------------------------


def equation(title: str, *lines: dict, formula: str | None = None, note: str | None = None) -> dict:
    """An equation block, for building one before deciding where it goes on a page. Each line
    is a term (`t`, `op`, `weight`), a `row` of terms side by side, or a `grid`; lines stack
    vertically."""
    return {"type": "equation", "title": title, "formula": formula, "note": note, "lines": list(lines)}


def t(
    tensor: str,
    label: str | None = None,
    *,
    rows: list[str] | None = None,
    cols: list[str] | None = None,
    fix: dict[str, int | str] | None = None,
    select: dict[str, str] | None = None,
    cell: float | None = None,
    values: bool = False,
    scale: str | None = None,
    wrap: int | None = None,
    norm: str | None = None,
    labels: bool | None = None,
) -> dict:
    """A tensor shown as a grid of colored squares.

    rows / cols: which axes go down / across (several axes are flattened, with
        gaps between groups). Defaults: first free axis down, the rest across.
    fix: pins axes to an index, or to a selection variable: "$pos" (the focused
        query token), "$kpos" (the focused key token) or "$head". Inside a "cache"
        block, "$j" is the cached token of the row being drawn.
    select: which selection variable clicking an axis sets, overriding the
        default (pos/q_pos → $pos, k_pos → $kpos, head → $head).
    cell: preferred square size in px. values: print numbers beside squares.
    scale: terms in one equation with the same scale name share a color scale.
    wrap: wrap columns every `wrap` squares instead of fitting the width.
    norm: "row" colors each row on its own scale.
    labels: False hides the row labels (e.g. when the term to its left already shows them).
    """
    term: dict[str, Any] = {"tensor": tensor}
    for k, v in dict(label=label, rows=rows, cols=cols, fix=fix, select=select, cell=cell, scale=scale, wrap=wrap, norm=norm, labels=labels).items():
        if v is not None:
            term[k] = v
    if values:
        term["values"] = True
    return term


def op(symbol: str) -> dict:
    """An operator or connective between terms, e.g. "+", "⊙", "= ", "÷ 8"."""
    return {"op": symbol}


def weight(name: str, shape: list[int], note: str | None = None) -> dict:
    """A parameter matrix that is used but not shipped to the page (it would be too large)."""
    w: dict[str, Any] = {"weight": name, "shape": list(shape)}
    if note:
        w["note"] = note
    return w


def row(*terms: dict) -> dict:
    """Terms side by side, wrapping onto more lines when the page is narrow."""
    return {"row": list(terms)}


def grid(*rows: list[dict | None]) -> dict:
    """Terms in aligned columns (None leaves a cell empty), e.g. to put a result
    directly under the matrix it was computed from."""
    return {"grid": [list(r) for r in rows]}
