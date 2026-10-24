"""Round-trip a small Trace through the writer and decode it the way the viewer does."""

import base64
import json
import re

import numpy as np

from actviz.trace import Trace, op, t
from actviz.writer import write_trace


def _payload(path):
    m = re.fullmatch(r'ACTVIZ\.receive\("(\w+)","([^"]*)",(.*)\);\n', path.read_text(), re.S)
    return m.group(1), m.group(2), json.loads(m.group(3))


def test_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    big = rng.normal(size=(3, 2000)).astype(np.float32)  # > F32_MAX_ELEMENTS → stored as f16
    small = rng.normal(size=(3, 4)).astype(np.float32)
    masked = np.where(np.tril(np.ones((3, 3), bool)), 1.0, -np.inf).astype(np.float32)

    tr = Trace(model={"name": "toy"}, prompt="a b c", tokens=[{"id": i, "text": s} for i, s in enumerate("abc")], axes={"pos": {"labels": "tokens"}})
    tr.add("big", big, ["pos", "d"], chunk="c1")
    tr.add("small", small, ["pos", "e"], chunk="c1")
    tr.add("masked", masked, ["q_pos", "k_pos"], chunk="c2")
    tr.alias("big0", "big", {"pos": 0})
    tr.section("s", "Section").page("p", "Page").equation("eq", t("big"), op("+"), t("small", fix={"pos": "$pos"}))
    out = write_trace(tr, tmp_path, "toy", "Toy")

    kind, key, manifest = _payload(out / "manifest.js")
    assert (kind, key) == ("manifest", "toy")
    meta = manifest["tensors"]
    assert meta["big"]["dtype"] == "f16" and meta["small"]["dtype"] == "f32"
    assert meta["big0"]["of"] == "big" and meta["big0"]["at"] == {"pos": 0}
    assert meta["masked"]["stats"]["masked"] == 3

    for tid, ref, tol in (("big", big, 2e-3), ("small", small, 0), ("masked", masked, 0)):
        m = meta[tid]
        _, _, b64 = _payload(out / manifest["chunks"][m["chunk"]]["file"])
        raw = base64.b64decode(b64)
        assert m["offset"] % 8 == 0
        dt = "<f2" if m["dtype"] == "f16" else "<f4"
        got = np.frombuffer(raw, dt, count=ref.size, offset=m["offset"]).astype(np.float32).reshape(ref.shape)
        np.testing.assert_allclose(got, ref, rtol=tol, atol=tol)

    _, _, index = _payload(tmp_path / "traces.js")
    assert [e["id"] for e in index] == ["toy"]
