# Direct LLM weight-space programming pilot

This is the first actual LLM weight-editing experiment in this repository. The
earlier numerical adaptation studies are baselines, not evidence about LLM WSP.

## Model, access, and spending

Use Claude Opus 5.5 (`claude-opus-5-5`), adaptive thinking, medium effort,
maximum 4,096 output tokens. The independently called model receives only a
small network's architecture, numerical weights, input/output meanings, a
physical-family prior, and an edit request. No tools, source code, probes,
training examples, forward passes, test labels, or feedback are supplied.
Each stateless call returns explicit numerical parameter replacements.

Official model availability was checked using the API. Published standard rates
checked 2026-11-10: $4/million input tokens and $20/million output tokens.
https://platform.claude.com/docs/en/about-claude/pricing
https://platform.claude.com/docs/en/api/messages/create

A shared persistent ledger caps development plus pilot generation at $3 using
conservative pre-request reservations and returned token usage. No automatic
generation retries. Credentials are read only from the environment and never
written. Only public final text, usage, model identity, and request metadata are
archived; thinking blocks are omitted. Token-counting precedes generation.

## Networks and tasks

Train one-hidden-layer 3-12-1 ReLU networks on pendulum-family accelerations
f=-k sin(q)-c v-a v|v|+g u with positive, undisclosed coefficients. Serialize
weights to eight decimal places and evaluate the serialized network itself.
Compare each learned network with an equivalent network containing two extra
zero-output hidden units. That spare capacity preserves the original function.
Original and permuted/positively rescaled representations are paired controls.

Three requests:
1. Action gain: implement f'(q,v,u)=f(q,v,1.5u). This is an exact architectural
   arithmetic control, implementable by scaling one input-weight column.
2. Known added drag: implement f'(x)=f(x)-0.2v. Spare ReLU pairs permit an exact
   compiled correction. Dense networks need not support an exact same-width edit.
3. Semantic damping: infer the unknown linear damping coefficient c from raw
   weights and double it, preserving the quadratic drag/restoring/action terms.
   Desired functional change is -c*v relative to the serialized source network.

The model reports its c estimate separately. Distinguish coefficient-reading
error from the error in compiling the model's own estimated correction.
Approximate trained networks do not identify a physical coefficient perfectly;
a source-query library-fit baseline quantifies this ambiguity.

## Planned pilot and controls

One development call on an unrelated source seed checks transport/schema. Freeze
the prompt and code before 24 pilot calls: two new source seeds (90,91), two
capacity conditions, two equivalent representations, and three tasks. These are
24 prompts derived from only two independently trained sources, not 24 independent
capability replications. A budget stop or API error remains visible in the report.

No result-driven prompt edits or regeneration of failed pilot responses. Invalid,
empty, truncated, or out-of-range patches count as failures. Patch application is
atomic and accepts only finite, bounded replacements of existing tensor entries.

References: unchanged model; deterministic action-column compiler; exact spare
ReLU compiler when applicable; a query-derived coefficient estimate plus spare
compiler; and same-architecture numerical head fitting on 128 synthetic edit
targets. Oracle coefficient references are explicitly marked as such.

Evaluate 4,096 held-out points and a larger extrapolation cube, immediate
preservation on zero-effect sets, source approximation error, parameter count
changed, and normalized requested-change error. Predefined success tolerances:
action control NRMSE<=1e-4 and collateral RMSE<=1e-6; known drag NRMSE<=0.05;
semantic drag NRMSE<=0.10; both drag tasks collateral RMSE<=0.01. Report raw
metrics and execution failures rather than hiding them behind a success rate.

No claim of novelty, broad reliability, or large-model scalability follows from
this tiny pilot. The purpose is to finally test direct raw-weight reasoning and
numerical editing, and identify a defensible next research question.
