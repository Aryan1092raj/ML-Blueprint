# Contributing

Welcome. This page is the whole process. It should take three minutes to read.

If anything here is confusing, that is our fault, not yours. Open an issue and say so.

## The only rule that matters

**You should be able to explain what you submitted.**

Not perfectly, and not on the first try. But if a reviewer asks "why did you do it this way?",
you should have an answer. That is what we are all here to build.

## First time contributing to anything?

That is fine, and this is a good place to start. Do this:

1. Read [docs/REPOSITORY_GUIDE.md](docs/REPOSITORY_GUIDE.md) if GitHub is new to you.
2. Set up the project with [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).
3. Find an issue labelled `good-first-issue` and comment on it saying you want it.
4. Ask questions in the issue. Asking early is normal here, not a sign you are struggling.

## Setup

```bash
git clone https://github.com/ACM-IIT-Mandi/ML-Blueprint.git
cd ML-Blueprint
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## What you can work on

| Label | What it means |
|---|---|
| `good-first-issue` | Small and scoped, with someone assigned to help you |
| `docs` | Explaining an algorithm in words, or fixing a confusing page |
| `implementation` | Writing an algorithm |
| `tests` | Adding cases to an existing test file |
| `visualisation` | Making an algorithm show its steps in the visualiser |
| `bug` | Something is broken |

Comment on an issue to claim it. If you want to work on something that has no issue yet, open
one first so nobody duplicates your work.

**Adding a new algorithm?** Open a "New algorithm" issue first and wait for a thumbs up. It
usually takes a day. This is only so two people do not write k-means in the same week.

## How to submit work

1. Make a branch: `git checkout -b algo/decision-tree` (or `fix/...`, `docs/...`).
2. Write your code. The layout for an algorithm folder is in
   [docs/ALGORITHM_TEMPLATE.md](docs/ALGORITHM_TEMPLATE.md).
3. Check it before pushing:

   ```bash
   ruff format .
   ruff check . --fix
   pytest
   ```

4. Push and open a pull request. Fill in the template, especially the part asking you to
   explain your work in your own words.
5. Wait for review. Expect questions. Questions are not criticism.

Open a **draft** pull request early if you want feedback halfway through. We would rather help
you at 50% than have you redo things at 100%.

## Things that trip people up

- **Do not commit Jupyter notebooks.** They make unreadable diffs. Examples go in `.py` files.
- **We do not copy scikit-learn.** Where it has the same algorithm, we use it in tests to
  check our answers. Plenty of what we're building, like the deep learning side, has no
  scikit-learn version at all, that's fine, see docs/TESTING.md for what to do instead.
- **No TensorFlow.** Plain Python and NumPy always. PyTorch only where it genuinely adds
  something.
- **Write derivations in your own words.** Copying from a textbook is a copyright problem, and
  you learn nothing from it.

## Reporting a bug

Open a bug issue with a short piece of code that shows the problem. Five lines of code beats
five paragraphs of description.

## Questions

Open a [Discussion](https://github.com/ACM-IIT-Mandi/ML-Blueprint/discussions), or ask in the
ACM channels. Nobody here will mind a basic question.

Everything you do here is covered by our [Code of Conduct](CODE_OF_CONDUCT.md).
