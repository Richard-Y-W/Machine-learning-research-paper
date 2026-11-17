"""Weights-only search for successor heads (Gould et al. 2023) in Qwen3-0.6B-Base.

For head h in layer l: OV_h = W_O[:, h] W_V[kv(h)]. Input: RMSNorm_l(h_l(x)) at the letter's own position
in "ENC: x ->". Output read through final norm + unembedding restricted to the 26 letters.
succ_score = fraction of letters a..y whose top letter is succ(x); copy_score = top letter is x.
"""
import json, string, torch
from .pilot import Skill, LETTERS, succ, PRACTICE, HELD

def main():
    sk = Skill(); m = sk.model; P = sk.params; L = sk.n_layers; HD = 128
    texts = [f"ENC: {c} ->" for c in LETTERS]
    enc = [sk.tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    pos = [e.index(sk.tok(" " + c, add_special_tokens=False)["input_ids"][0]) for e, c in zip(enc, LETTERS)]
    ids, mask, last = sk._batch(texts)
    with torch.inference_mode():
        out = m(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
    WU = (P["lm_head.weight"] if "lm_head.weight" in P else P["model.embed_tokens.weight"]).float()
    lt = torch.tensor([sk.ans[c] for c in string.ascii_lowercase]); WUl = WU[lt]           # 26 x d
    gf = P["model.norm.weight"].float()
    rows = []
    with torch.inference_mode():
        for l in range(L):
            h = out.hidden_states[l][torch.arange(len(texts)), torch.tensor(pos)].float()
            x = h / h.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt() * P[f"model.layers.{l}.input_layernorm.weight"].float()
            Wv = P[f"model.layers.{l}.self_attn.v_proj.weight"].float(); Wo = P[f"model.layers.{l}.self_attn.o_proj.weight"].float()
            for hd in range(16):
                kv = hd // 2
                v = x @ Wv[kv * HD:(kv + 1) * HD].T                      # n x 128
                o = v @ Wo[:, hd * HD:(hd + 1) * HD].T                   # n x d
                logits = (o * gf) @ WUl.T                                # n x 26 (direction only; norm scale irrelevant to argmax)
                top = logits.argmax(-1)
                tgt = torch.tensor([string.ascii_lowercase.index(succ(c)) for c in LETTERS])
                self_ = torch.tensor([string.ascii_lowercase.index(c) for c in LETTERS])
                hi = torch.tensor([LETTERS.index(c) for c in HELD])
                rows.append(dict(layer=l, head=hd, succ=float((top == tgt).float().mean()), copy=float((top == self_).float().mean()),
                                 succ_held=float((top[hi] == tgt[hi]).float().mean())))
    rows.sort(key=lambda r: -r["succ"])
    json.dump(rows, open("results/skill/theory/successor_heads.json", "w"), indent=1)
    print("top successor heads (succ score over 25 letters, held-out-letter succ, copy score):")
    for r in rows[:10]:
        print(f"L{r['layer']}h{r['head']}: succ {r['succ']:.2f} held {r['succ_held']:.2f} copy {r['copy']:.2f}")
    print("chance ~ 1/26 = 0.04")

if __name__ == "__main__":
    main()
