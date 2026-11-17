"""Edit language for LLM-authored (and random) mutations, and the genome.

A *proposal* is JSON from the editor (or the random sampler). It is *resolved*
against the parent model and a RefBank into concrete tensors. A resolved
mutation is replayable: applying the same resolved ops, in order, to the same
weights gives identical tensors. A genome is the ordered list of resolved
mutations accepted along a lineage, applied to the original checkpoint.
"""
import re

import torch

MAX_OPS = 8
MAX_SET_ENTRIES = 256
MAX_REL_CHANGE = 0.50       # per touched tensor, ||dW||_F / ||W||_F for one mutation
MAX_ALPHA = 0.5             # rank-1 coefficient relative to the tensor's top singular value
FACTOR_RANGE = (0.0, 3.0)

_PARTS = {
    "attn.q": "self_attn.q_proj.weight", "attn.k": "self_attn.k_proj.weight",
    "attn.v": "self_attn.v_proj.weight", "attn.o": "self_attn.o_proj.weight",
    "attn.qnorm": "self_attn.q_norm.weight", "attn.knorm": "self_attn.k_norm.weight",
    "mlp.gate": "mlp.gate_proj.weight", "mlp.up": "mlp.up_proj.weight", "mlp.down": "mlp.down_proj.weight",
    "ln1": "input_layernorm.weight", "ln2": "post_attention_layernorm.weight",
}
HEAD_DIM = 128


class InvalidEdit(ValueError):
    pass


def param_name(alias):
    if alias == "final_norm":
        return "model.norm.weight"
    m = re.fullmatch(r"L(\d+)\.(.+)", alias)
    if not m or m.group(2) not in _PARTS:
        raise InvalidEdit(f"unknown tensor alias {alias!r}")
    return f"model.layers.{int(m.group(1))}.{_PARTS[m.group(2)]}"


def all_aliases(n_layers):
    return [f"L{i}.{p}" for i in range(n_layers) for p in _PARTS] + ["final_norm"]


def head_index(alias, head, shape):
    """Indices and axis for one attention head of a q/k/v (rows) or o (cols) tensor."""
    part = alias.split(".", 1)[1]
    if part in ("attn.q", "attn.k", "attn.v"):
        n = shape[0] // HEAD_DIM; axis = "row"
    elif part == "attn.o":
        n = shape[1] // HEAD_DIM; axis = "col"
    else:
        raise InvalidEdit(f"'head' only applies to attn.q/k/v/o, not {alias}")
    if not 0 <= head < n:
        raise InvalidEdit(f"head {head} out of range 0..{n - 1} for {alias}")
    return axis, torch.arange(head * HEAD_DIM, (head + 1) * HEAD_DIM)


class RefBank:
    """Named vectors an editor can reference in rank-1 updates.

    vectors: dict name -> 1-D tensor (e.g. 'err:L12' residual error direction).
    Dynamic refs: 'sv:<alias>:<k>:left|right', 'unit:<i>', 'tok:<text>', 'letter:<A-E>'.
    """

    def __init__(self, params, tok=None, vectors=None):
        self.params = params; self.tok = tok; self.vectors = dict(vectors or {})

    def get(self, ref, dim):
        if ref in self.vectors:
            v = self.vectors[ref]
        elif ref.startswith("sv:"):
            try:
                _, alias, k, side = ref.split(":")
                k = int(k)
            except ValueError:
                raise InvalidEdit(f"bad sv ref {ref!r}; use sv:<tensor>:<k>:left|right")
            W = self.params[param_name(alias)]
            if W.ndim != 2 or not 0 <= k < 8 or side not in ("left", "right"):
                raise InvalidEdit(f"bad sv ref {ref!r}")
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            v = U[:, k] if side == "left" else Vh[k]
        elif ref.startswith("unit:"):
            i = int(ref.split(":")[1])
            if not 0 <= i < dim:
                raise InvalidEdit(f"{ref} out of range for dim {dim}")
            v = torch.zeros(dim); v[i] = 1.0
        elif ref.startswith(("tok:", "letter:")) and self.tok is not None:
            text = ref.split(":", 1)[1]
            text = " " + text if ref.startswith("letter:") else text
            ids = self.tok(text, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                raise InvalidEdit(f"{ref} is not a single token")
            v = self.params["model.embed_tokens.weight"][ids[0]]
        else:
            raise InvalidEdit(f"unknown vector ref {ref!r}")
        v = v.detach().float().clone()
        if v.numel() != dim:
            raise InvalidEdit(f"ref {ref!r} has dim {v.numel()}, tensor needs {dim}")
        n = v.norm()
        if not torch.isfinite(n) or n == 0:
            raise InvalidEdit(f"ref {ref!r} is zero or non-finite")
        return v / n


def top_sigma(W):
    return float(torch.linalg.matrix_norm(W, ord=2)) if W.ndim == 2 else float(W.abs().max())


def resolve(proposal_ops, params, bank):
    """Turn proposal ops (JSON dicts) into concrete replayable ops. Raises InvalidEdit."""
    if not isinstance(proposal_ops, list) or not 1 <= len(proposal_ops) <= MAX_OPS:
        raise InvalidEdit(f"a mutation needs 1..{MAX_OPS} ops")
    out, n_set = [], 0
    for op in proposal_ops:
        kind = op.get("op"); alias = op.get("tensor", "")
        name = param_name(alias); W = params[name]
        if kind in ("scale", "ablate"):
            factor = 0.0 if kind == "ablate" else float(op.get("factor", 1.0))
            if not FACTOR_RANGE[0] <= factor <= FACTOR_RANGE[1]:
                raise InvalidEdit(f"factor {factor} outside {FACTOR_RANGE}")
            if op.get("head") is not None:
                axis, idx = head_index(alias, int(op["head"]), W.shape)
            elif W.ndim == 1:
                axis = "row"; idx = op.get("index")
            else:
                axis = op.get("axis", "row"); idx = op.get("index")
                if axis not in ("row", "col", "all"):
                    raise InvalidEdit(f"axis must be row|col|all, got {axis!r}")
            if axis != "all":
                if idx is None or len(idx) == 0:
                    raise InvalidEdit("scale/ablate needs 'index' (list) or 'head' unless axis='all'")
                idx = torch.as_tensor(idx, dtype=torch.long)
                lim = W.shape[0] if axis == "row" else W.shape[1]
                if idx.min() < 0 or idx.max() >= lim:
                    raise InvalidEdit(f"index out of range 0..{lim - 1} for {alias} axis {axis}")
            else:
                idx = None
            out.append(dict(op="scale", tensor=alias, axis=axis, index=idx, factor=factor))
        elif kind == "rank1":
            if W.ndim != 2:
                raise InvalidEdit(f"rank1 needs a matrix; {alias} is 1-D")
            alpha = float(op.get("alpha", 0.0))
            if not 0 < abs(alpha) <= MAX_ALPHA:
                raise InvalidEdit(f"|alpha| must be in (0, {MAX_ALPHA}]")
            u = bank.get(op.get("u", ""), W.shape[0]); v = bank.get(op.get("v", ""), W.shape[1])
            coef = alpha * top_sigma(W)
            out.append(dict(op="rank1", tensor=alias, u=u, v=v, coef=coef, alpha=alpha,
                            u_ref=op.get("u"), v_ref=op.get("v")))
        elif kind == "set":
            entries = op.get("entries") or []
            n_set += len(entries)
            if not entries or n_set > MAX_SET_ENTRIES:
                raise InvalidEdit(f"set needs 1..{MAX_SET_ENTRIES} entries in total")
            rows, cols, vals = [], [], []
            for e in entries:
                if W.ndim == 1:
                    if len(e) != 2: raise InvalidEdit("1-D set entry is [index, value]")
                    rows.append(int(e[0])); cols.append(0); vals.append(float(e[1]))
                else:
                    if len(e) != 3: raise InvalidEdit("2-D set entry is [row, col, value]")
                    rows.append(int(e[0])); cols.append(int(e[1])); vals.append(float(e[2]))
            rows = torch.tensor(rows); cols = torch.tensor(cols); vals = torch.tensor(vals)
            if rows.min() < 0 or rows.max() >= W.shape[0] or (W.ndim == 2 and (cols.min() < 0 or cols.max() >= W.shape[1])):
                raise InvalidEdit(f"set index out of range for {alias} {tuple(W.shape)}")
            if not torch.isfinite(vals).all() or vals.abs().max() > 10 * float(W.abs().max()):
                raise InvalidEdit("set value non-finite or >10x the tensor's max |w|")
            out.append(dict(op="set", tensor=alias, rows=rows, cols=cols if W.ndim == 2 else None, values=vals))
        elif kind == "revert":
            out.append(dict(op="revert", tensor=alias))
        else:
            raise InvalidEdit(f"unknown op {kind!r}")
    return out


@torch.no_grad()
def apply_op(params, op, original):
    W = params[param_name(op["tensor"])]
    k = op["op"]
    if k == "scale":
        if op["axis"] == "all": W.mul_(op["factor"])
        elif op["axis"] == "row": W[op["index"]] *= op["factor"]
        else: W[:, op["index"]] *= op["factor"]
    elif k == "rank1":
        W.add_(op["coef"] * torch.outer(op["u"], op["v"]))
    elif k == "set":
        if op["cols"] is None: W[op["rows"]] = op["values"]
        else: W[op["rows"], op["cols"]] = op["values"]
    elif k == "revert":
        W.copy_(original[param_name(op["tensor"])])
    else:
        raise InvalidEdit(k)


class Applied:
    """Context: apply a resolved mutation in place, validate, restore on exit unless kept."""

    def __init__(self, params, mutation, original):
        self.params, self.mutation, self.original = params, mutation, original
        self.keep = False

    def __enter__(self):
        names = {param_name(op["tensor"]) for op in self.mutation}
        self.backup = {n: self.params[n].detach().clone() for n in names}
        for op in self.mutation:
            apply_op(self.params, op, self.original)
        self.rel = {}
        for n, old in self.backup.items():
            new = self.params[n]
            if not torch.isfinite(new).all():
                self._restore(); raise InvalidEdit(f"non-finite weights in {n}")
            self.rel[n] = float((new - old).norm() / old.norm().clamp_min(1e-12))
        bad = {n: r for n, r in self.rel.items() if r > MAX_REL_CHANGE and not self._is_revert(n)}
        if bad:
            self._restore(); raise InvalidEdit(f"relative change too large: {bad}")
        return self

    def _is_revert(self, n):
        return any(op["op"] == "revert" and param_name(op["tensor"]) == n for op in self.mutation)

    def _restore(self):
        with torch.no_grad():
            for n, old in self.backup.items():
                self.params[n].copy_(old)

    def __exit__(self, *exc):
        if not self.keep or exc[0] is not None:
            self._restore()
        return False


def replay(params, genome, original):
    """Apply every resolved mutation of a genome in order (params should start at original)."""
    for mutation in genome:
        for op in mutation:
            apply_op(params, op, original)


def describe(mutation):
    """Compact human/LLM-readable summary of a resolved mutation."""
    parts = []
    for op in mutation:
        if op["op"] == "scale":
            where = "all" if op["axis"] == "all" else f"{op['axis']}[{_short(op['index'])}]"
            parts.append(f"scale {op['tensor']} {where} x{op['factor']:.3g}")
        elif op["op"] == "rank1":
            parts.append(f"rank1 {op['tensor']} u={op['u_ref']} v={op['v_ref']} alpha={op['alpha']:+.3g}")
        elif op["op"] == "set":
            parts.append(f"set {op['tensor']} {len(op['values'])} entries")
        else:
            parts.append(f"revert {op['tensor']}")
    return "; ".join(parts)


def _short(idx):
    v = idx.tolist()
    if len(v) > 1 and v == list(range(v[0], v[-1] + 1)):
        return f"{v[0]}:{v[-1] + 1}"
    return ",".join(map(str, v[:6])) + ("..." if len(v) > 6 else "")


def min_layer(mutation, n_layers):
    """Lowest decoder layer a mutation touches (n_layers if only final_norm)."""
    ls = [n_layers if op["tensor"] == "final_norm" else int(op["tensor"][1:].split(".")[0]) for op in mutation]
    return min(ls)


def genome_delta(params_now, original, names=None):
    """Dict name -> (W_now - W_orig) for tensors that changed."""
    out = {}
    for n in (names or original):
        d = params_now[n] - original[n]
        if d.abs().max() > 0:
            out[n] = d
    return out
