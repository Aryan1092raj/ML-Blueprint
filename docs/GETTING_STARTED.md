# Getting started

Five minutes, start to finish.

## You need

Python 3.10 or newer (`python --version` to check), and git.

## Setup

```bash
git clone https://github.com/ACM-IIT-Mandi/ML-Blueprint.git
cd ML-Blueprint
```

Make a virtual environment. This keeps the project's libraries separate from the rest of your
system, which saves you a lot of pain later.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

Install it:

```bash
pip install -e ".[dev]"
```

The `-e` means your edits take effect straight away, with no reinstall. Check it worked:

```bash
pytest
```

## Commands you will use

```bash
pytest                                   # run the tests
pytest mlblueprint/cluster/kmeans        # run one algorithm's tests

ruff format .                            # tidy the formatting
ruff check . --fix                       # find and fix style problems

streamlit run apps/visualiser/app.py     # open the visualiser
```

Run the first three before you push. The automated checks run the same things, so doing it
locally saves you a round trip.

## What got installed

NumPy is the only thing the library really needs. The `[dev]` part adds scikit-learn (we use it
to check our answers), matplotlib, Streamlit, and the test and formatting tools.

PyTorch is not included, because it is a big download and most work does not need it. If you
are working on a `torch_impl.py`:

```bash
pip install -e ".[dev,torch]"
```

## Where to go next

| If you want to | Read |
|---|---|
| Understand what all these files are | [REPOSITORY_GUIDE.md](REPOSITORY_GUIDE.md) |
| Contribute something | [../CONTRIBUTING.md](../CONTRIBUTING.md) |
| Add an algorithm | [ALGORITHM_TEMPLATE.md](ALGORITHM_TEMPLATE.md) |
| Write code or docs | [STYLE_GUIDE.md](STYLE_GUIDE.md) |
| Write tests | [TESTING.md](TESTING.md) |
| Add a visualisation | [VISUALISATION.md](VISUALISATION.md) |
| Know why the layout is like this | [ARCHITECTURE.md](ARCHITECTURE.md) |

## If something breaks

**`ModuleNotFoundError: No module named 'mlblueprint'`**
Your virtual environment is not active. Run the `source .venv/bin/activate` line again.

**`pytest` finds no tests**
You are not in the project folder. `cd` back to it.

**The visualiser says there are no panels**
That is correct for now. There are no algorithms yet, so there is nothing to show.

**Anything else**
Open a [Discussion](https://github.com/ACM-IIT-Mandi/ML-Blueprint/discussions). If these
instructions did not work for you, that is a bug in this page and worth reporting.
