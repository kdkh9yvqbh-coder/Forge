# Forge

Forge is an adversarially validated improvement engine.

It does not write code from vibes. It analyzes a supported artifact, transforms it under explicit objectives, attacks its own conclusions, and reports whether the improvement is real.

## What Forge does

Forge supports four main workflows:

- **optimize**: improve efficiency while preserving behavior
- **repair**: recover correctness or intended behavior
- **harden**: improve worst-case robustness under hostile inputs
- **improve(function, tests=...)**: lift a restricted Python function into a structural search space and improve it under test pressure

Forge ships with four proof fronts:

- sorting
- pathfinding
- regex symbolic transformation
- restricted Python Autolift

## Core ideas

Forge combines:

- planner-guided search
- motif graph memory
- promoted operators and promoted sequences
- Counter-Forge adversarial attack
- post-attack recovery
- forensics and causal traces

That means Forge does not just produce a winner. It also asks whether the winner survives pressure.

## Why this exists

Most coding tools try to generate code for you.

Forge is built for a different philosophy:

- start from something real
- transform it structurally
- test it on proof and hidden cases
- attack it with Counter-Forge
- recover if needed
- explain why the result is better

It is closer to a cross-examiner than a ghostwriter.

## Current release

This repo currently centers on the standalone public release script:

`forge_o1_8_1.py`

The release includes:

- standalone CLI
- Python API
- multi-domain proof demos
- restricted Autolift
- markdown report generation

## Quick start

Run the full demo:

```bash
python forge_o1_8_1.py demo
```

List domains and commands:

```bash
python forge_o1_8_1.py list-domains
```

Run the Autolift demo:

```bash
python forge_o1_8_1.py autolift-demo
```

Run the regex proof-domain demo:

```bash
python forge_o1_8_1.py regex-domain-demo
```

## Python API

You can import the release file directly:

```python
from forge_o1_8_1 import optimize, repair, harden, improve
```

Example:

```python
from forge_o1_8_1 import improve

def count_positive(nums):
    total = 0
    for x in nums:
        if x > 0:
            total += 1
    return total

tests = [
    (([],), 0),
    (([1, -1, 2, 0],), 2),
    (([-5, -4],), 0),
]

result = improve(count_positive, tests=tests)
print(result.improved_source)
```

## CLI commands

The public release exposes:

- `demo`
- `autolift-demo`
- `regex-domain-demo`
- `list-domains`

## What “restricted Autolift” means

Autolift is intentionally constrained.

Forge can analyze and improve a restricted subset of Python functions that are:

- deterministic
- mostly pure
- structurally analyzable
- free of I/O, subprocess calls, dynamic execution, and similar runtime chaos

This is deliberate. The goal is to keep improvement auditable and testable, not magical.

## Output and evidence

Forge produces evidence, not just candidates. Depending on the run, that can include:

- proof metrics
- hidden metrics
- attack metrics
- planner traces
- motif chains
- recovery information
- before/after source
- diff previews
- markdown reports

## What Forge is not

Forge is not:

- an LLM code generator
- an unconstrained AutoML wrapper
- a benchmark memorizer
- a universal optimizer for arbitrary Python
- a replacement for engineering judgment

## Project position

The simplest way to describe Forge is:

> Forge is an adversarially validated improvement engine. It does not generate code from vibes. It analyzes, transforms, attacks, and explains whether an improvement is real.

## Files

Recommended starting points:

- `forge_o1_8_1.py`
- `forge_o1_8_1_report.md`

## Status

The public release is the first release-shaped build.

It already includes the full engine stack, but it is still evolving. The next likely work is packaging polish, repo structure cleanup, and stronger public-facing examples.

## License

Choose the license that matches how you want the project used before publishing.
