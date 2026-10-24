"""Serializing a Trace into files the static viewer can load.

Layout of an output directory (default `viewer/data/`):

    traces.json / traces.js      index of every exported trace
    <trace-id>/manifest.js       model info, tokens, tensor metadata, pages
    <trace-id>/<chunk>.js        tensor data, base64-encoded
    <trace-id>/labels-<name>.js  large label lists (e.g. the vocabulary)

Every file is a script calling `ACTVIZ.receive(kind, key, payload)` rather than
JSON fetched with XHR, so the page also works when opened straight from disk
(file:// URLs can't be fetched, but they can be loaded as <script>s). Chunks
are loaded lazily, only when a page needs one of their tensors.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from .trace import Trace

# Tensors this small are stored as float32; everything else as float16, which
# halves the download and is plenty for color plus a 3-4 significant digit readout.
F32_MAX_ELEMENTS = 4096
FORMAT = "actviz/1"


def _js(kind: str, key: str, payload) -> str:
    body = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    return f"ACTVIZ.receive({json.dumps(kind)},{json.dumps(key)},{body});\n"


def _stats(data: np.ndarray, kind: str, scale: float | None) -> dict:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return {"min": 0, "max": 0, "absmax": 1, "robust": 1}
    mag = np.abs(finite) if kind == "signed" else np.maximum(finite, 0)
    absmax = float(mag.max())
    robust = float(scale) if scale is not None else float(np.percentile(mag, 99))
    if not robust > 0:
        robust = absmax if absmax > 0 else 1.0
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "absmax": absmax if absmax > 0 else 1.0,
        "robust": robust,
        "masked": int(data.size - finite.size),
    }


def write_trace(trace: Trace, out_dir: Path, trace_id: str, title: str) -> Path:
    trace.validate()
    out_dir = Path(out_dir)
    tdir = out_dir / trace_id
    tdir.mkdir(parents=True, exist_ok=True)
    for old in tdir.glob("*.js"):
        old.unlink()

    tensors_meta: dict[str, dict] = {}
    chunks: dict[str, bytearray] = {}
    for rec in trace.tensors.values():
        meta = {
            "shape": list(rec.data.shape),
            "axes": rec.axes,
            "kind": rec.kind,
            "label": rec.label,
            "stats": _stats(rec.data, rec.kind, rec.scale),
        }
        if rec.desc:
            meta["desc"] = rec.desc
        if rec.percent:
            meta["percent"] = True
        if rec.of:
            meta["of"] = rec.of
            meta["at"] = rec.at
        else:
            buf = chunks.setdefault(rec.chunk, bytearray())
            buf.extend(b"\0" * (-len(buf) % 8))
            finite = rec.data[np.isfinite(rec.data)]
            fits_f16 = finite.size == 0 or float(np.abs(finite).max()) < 65000
            dtype = "f32" if rec.data.size <= F32_MAX_ELEMENTS or not fits_f16 else "f16"
            arr = rec.data.astype("<f4" if dtype == "f32" else "<f2")
            meta.update(chunk=rec.chunk, offset=len(buf), dtype=dtype)
            buf.extend(arr.tobytes(order="C"))
        tensors_meta[rec.id] = meta

    chunk_meta = {}
    for name, buf in chunks.items():
        b64 = base64.b64encode(bytes(buf)).decode("ascii")
        (tdir / f"{name}.js").write_text(_js("chunk", f"{trace_id}/{name}", b64))
        chunk_meta[name] = {"file": f"{name}.js", "bytes": len(buf)}

    label_meta = {}
    for name, labels in trace.labels.items():
        (tdir / f"labels-{name}.js").write_text(_js("labels", f"{trace_id}/{name}", labels))
        label_meta[name] = {"file": f"labels-{name}.js", "count": len(labels)}

    manifest = {
        "format": FORMAT,
        "id": trace_id,
        "title": title,
        "model": trace.model,
        "prompt": trace.prompt,
        "tokens": trace.tokens,
        "axes": trace.axes,
        "tensors": tensors_meta,
        "chunks": chunk_meta,
        "labels": label_meta,
        "sections": [s.to_json() for s in trace.sections],
        "extra": trace.extra,
    }
    (tdir / "manifest.js").write_text(_js("manifest", trace_id, manifest))

    index_path = out_dir / "traces.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else []
    entry = {
        "id": trace_id,
        "title": title,
        "model": trace.model.get("display_name", trace.model.get("name")),
        "prompt": trace.prompt,
        "n_tokens": len(trace.tokens),
    }
    index = [e for e in index if e["id"] != trace_id] + [entry]
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")
    (out_dir / "traces.js").write_text(_js("traces", "", index))
    return tdir
