<div align="center">

# ML Blueprint

**Machine learning algorithms, written so you can actually understand them.**

A project by [ACM IIT Mandi](https://github.com/ACM-IIT-Mandi)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

</div>

---

## The idea

Most of us finish an ML course able to call `.fit()` but unable to say what happened inside it.

ML Blueprint is our attempt to fix that. For every algorithm we write down the maths, then the
code, then tests that prove the code matches the maths. All of it in one folder, so you can
read the whole story in one place.

It is a learning project. We are students, we are building it as we learn, and it is not
finished. That is the point.

## What an algorithm looks like here

```
mlblueprint/tree/decision_tree/
├── README.md          what it is, and how far along it is
├── scratch.py         plain Python, written to be read
├── numpy_impl.py      the fast version
├── docs/              intuition.md, derivation.md, complexity.md, references.md
├── tests/             proof that it works
├── example.py         a script you can run
└── viz.py             optional, adds it to the visualiser
```

You do not have to write all of that. Adding one piece to an algorithm someone else started is
a completely normal contribution.

## Layout

```
mlblueprint/          the library
├── core/             shared building blocks used by every algorithm
├── linear/           linear and logistic regression
├── tree/             decision trees and ensembles
├── cluster/          k-means, DBSCAN, GMM
├── decomposition/    PCA, SVD, t-SNE
├── svm/              support vector machines
├── probabilistic/    naive Bayes, HMMs
├── optim/            gradient descent and friends
└── neural/           autograd, layers, networks

apps/visualiser/      one app that shows algorithms running, step by step
docs/                 how to write things here
tests/                tests for the shared code
```

Right now the folders are mostly empty. We are setting up the foundation first so that the
first real algorithms have something solid to land on.

## Try it

```bash
git clone https://github.com/ACM-IIT-Mandi/ML-Blueprint.git
cd ML-Blueprint
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

More detail in [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).

## Want to help?

Yes please. You do not need to be good at ML yet, and you do not need to know GitHub.

- Never used GitHub? Start with [docs/REPOSITORY_GUIDE.md](docs/REPOSITORY_GUIDE.md). It
  explains what every file here is for, and there is a glossary at the end.
- Ready to contribute? [CONTRIBUTING.md](CONTRIBUTING.md) is the whole process, and it is
  short.
- Looking for a first task? Check the issues labelled `good-first-issue`.

Using AI tools is fine. Submitting code you cannot explain is not. See
[AI_POLICY.md](AI_POLICY.md).

## Who runs this

See [MAINTAINERS.md](MAINTAINERS.md). Be nice to each other:
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

MIT. See [LICENSE](LICENSE).
