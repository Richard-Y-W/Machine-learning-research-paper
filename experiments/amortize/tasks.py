"""Ten code-tracing program families (docs/amortize-editing-plan.md, section 4).
Every item: random variable names and values; the answer is computed by executing the program."""
import contextlib
import io
import random

PROMPT = "What does this Python program print? End your reply with `Answer: <number>`.\n```python\n{code}\n```"
NAMES = list("abcdefghkmnpqrstuvwxyz") + ["total", "count", "val", "num", "res", "tmp", "left", "right", "item", "acc"]
# Roles after stage 0a screening (plan amendment A1): F8 and F9 failed the screen (thinking accuracy < .80)
# and are reported separately; F7 moved from validation to test so the test set keeps two unseen structures.
DEV, VAL_HELD, TEST_HELD, BELOW_THRESHOLD = ["F1", "F2", "F3", "F4", "F5"], ["F6"], ["F7", "F10"], ["F8", "F9"]
DIRECT_SOLVABLE = ["F1", "F2"]      # answered correctly ~65% of the time with no thinking at all


def _n(rng, k):
    return rng.sample(NAMES, k)


def _v(rng, k, lo=10, hi=99):
    return rng.sample(range(lo, hi + 1), k)


def f1(rng):   # reassignment chain
    a, b, d, e = _n(rng, 4); x, y, z = _v(rng, 3)
    return [f"{a} = {x}", f"{b} = {a}", f"{a} = {z}", f"{d} = {y}", f"{e} = {d}", f"{d} = {b}", f"print({rng.choice([b, d, e, a])})"]


def f2(rng):   # swap through a temporary
    a, b, t, c = _n(rng, 4); x, y, z = _v(rng, 3)
    return [f"{a} = {x}", f"{b} = {y}", f"{c} = {z}", f"{t} = {a}", f"{a} = {b}", f"{b} = {t}", f"{c} = {a}", f"print({rng.choice([a, b, c])})"]


def f3(rng):   # augmented assignment
    x, y = _n(rng, 2); v0, w0 = _v(rng, 2); ks = [rng.randrange(2, 20) for _ in range(4)]
    return [f"{x} = {v0}", f"{y} = {w0}", f"{x} += {ks[0]}", f"{y} -= {ks[1]}", f"{x} -= {ks[2]}", f"{y} += {ks[3]}", f"print({rng.choice([x, y])})"]


def f4(rng):   # if / else on a comparison
    a, b, r = _n(rng, 3); x, y = _v(rng, 2); k = rng.randrange(2, 20); op = rng.choice([">", "<"])
    return [f"{a} = {x}", f"{b} = {y}", f"if {a} {op} {b}:", f"    {r} = {a}", "else:", f"    {r} = {b}", f"{r} = {r} + {k}", f"print({r})"]


def f5(rng):   # short for loop
    s, i = _n(rng, 2); s0 = rng.randrange(10, 60); n = rng.randrange(2, 5); k = rng.randrange(2, 10)
    return [f"{s} = {s0}", f"for {i} in range({n}):", f"    {s} += {k}", f"print({s})"]


def f6(rng):   # list index and update
    xs, y = _n(rng, 2); v = _v(rng, 4); i, j = rng.sample(range(3), 2)
    return [f"{xs} = [{v[0]}, {v[1]}, {v[2]}]", f"{xs}[{i}] = {v[3]}", f"{y} = {xs}[{j}]", f"{xs}[{j}] = {xs}[{i}]", f"print({xs}[{j}] + {y})"]


def f7(rng):   # dictionary lookup chain
    d, k = _n(rng, 2); keys = rng.sample(["p", "q", "r", "s", "t"], 3); v = _v(rng, 3)
    return [f'{d} = {{"{keys[0]}": {v[0]}, "{keys[1]}": {v[1]}, "{keys[2]}": {v[2]}}}', f'{k} = "{keys[1]}"', f'{d}["{keys[0]}"] = {d}[{k}]',
            f'{k} = "{keys[2]}"', f'print({d}["{keys[0]}"] + {d}[{k}])']


def _word(rng):
    return "".join(rng.choice("abcdefgh") for _ in range(rng.randrange(2, 6)))


def f8(rng):   # string concatenation and len
    s, t = _n(rng, 2)
    return [f'{s} = "{_word(rng)}"', f'{t} = {s} + "{_word(rng)}"', f"{s} = {t} + {s}", f"print(len({s}))"]


def f9(rng):   # small function call
    f, a, b = _n(rng, 3); body = rng.choice(["x + y", "x - y", "x + 2 * y", "2 * x + y"]); p, q, r = [rng.randrange(2, 15) for _ in range(3)]
    return [f"def {f}_fn(x, y):", f"    return {body}", f"{a} = {f}_fn({p}, {q})", f"{b} = {f}_fn({a}, {r})", f"print({b})"]


def f10(rng):  # tuple unpacking
    a, b, c = _n(rng, 3); x, y = _v(rng, 2, 2, 30)
    return [f"{a}, {b} = {x}, {y}", f"{a}, {b} = {b}, {a} + {b}", f"{c}, {a} = {a}, {b}", f"print({rng.choice([a, b, c])})"]


FAMILIES = dict(F1=f1, F2=f2, F3=f3, F4=f4, F5=f5, F6=f6, F7=f7, F8=f8, F9=f9, F10=f10)


def make(family, rng):
    lines = FAMILIES[family](rng); code = "\n".join(lines)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(code, {})
    return dict(family=family, q=PROMPT.format(code=code), a=buf.getvalue().strip(), kind="int", n_lines=len(lines), code=code)


def sample(families, n_each, seed):
    out = []
    for fam in families:
        rng = random.Random(f"{seed}|{fam}")
        out += [make(fam, rng) for _ in range(n_each)]
    return out
