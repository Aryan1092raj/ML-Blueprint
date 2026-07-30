# What every file here is for

New to GitHub? This page walks through the whole project folder by folder, in plain language.
There's a glossary of common terms at the bottom too.

## The fast version

| You want to | Open |
|---|---|
| Understand the project | `README.md` |
| Contribute something | `CONTRIBUTING.md` |
| Set up your computer | `docs/GETTING_STARTED.md` |
| Add an algorithm | `docs/ALGORITHM_TEMPLATE.md` |
| Find the actual code | `mlblueprint/` |

Everything else supports one of those five things.

## Files GitHub reads specially

A few files aren't just for people, GitHub itself reads them and changes how the website
behaves. That's why they need to be in exact spots with exact names.

| File | What it does |
|---|---|
| `README.md` | Shows automatically on the project's front page |
| `LICENSE` | Says who is allowed to use this code, and how |
| `CODE_OF_CONDUCT.md` | Adds a link to the community info on GitHub |
| `CONTRIBUTING.md` | GitHub links this when someone opens an issue or PR |
| `.github/PULL_REQUEST_TEMPLATE.md` | Fills in the box when someone opens a pull request |
| `.github/ISSUE_TEMPLATE/*.yml` | Turns "New issue" into a set of forms |
| `.github/CODEOWNERS` | Automatically asks the right people to review a change |
| `.github/workflows/ci.yml` | Runs tests automatically on every push |
| `.gitignore` | Tells git which files to never save |

Move or rename one of these and the feature just quietly stops working, with no error message.
Good to know before you go rearranging things.

## The root folder

**`README.md`**, the front page. What the project is, why it exists, how to get started.

**`LICENSE`**, ours is MIT. It means anyone can use, copy, or build on this code, including
for commercial projects, as long as they keep our name on it. Without this file, nobody is
legally allowed to reuse the code at all, even though it's public.

**`CONTRIBUTING.md`**, the whole process for contributing, in one short page.

**`CODE_OF_CONDUCT.md`**, how we treat each other, and what happens if someone doesn't.

**`AI_POLICY.md`**, our own addition. AI tools are fine, submitting code you can't explain is
not.

**`ROADMAP.md`**, what's built, what's next, roughly when.

**`MAINTAINERS.md`**, who's running the project and how to reach them.

**`pyproject.toml`**, the settings file for the Python project itself: its name, what
libraries it needs, and settings for our formatting and testing tools. This is what makes
`pip install -e .` and `import mlblueprint` work.

**`.gitignore`**, a list of files git should ignore, things like your virtual environment and
cache folders. Without it, those would get accidentally uploaded and clutter every change.

## `.github/`

Settings for GitHub itself, not for Python.

**`workflows/ci.yml`**, runs checks automatically every time someone pushes code: is it
formatted properly, do the tests pass, does it work with just the basic setup installed. Green
tick means all good, red cross means something needs fixing. This means a reviewer doesn't have
to manually check any of that themselves.

**`PULL_REQUEST_TEMPLATE.md`**, the checklist that appears when opening a pull request.

**`ISSUE_TEMPLATE/`**, turns "New issue" into a few simple forms instead of a blank box, one
for proposing an algorithm, one for bugs, one for docs problems.

**`CODEOWNERS`**, when someone changes certain files, it automatically tags the right people
to review it. See the [maintainers section](#adding-maintainers) below for how to set this up.

## `docs/`

Our own standards, written for contributors.

| File | For |
|---|---|
| `REPOSITORY_GUIDE.md` | This page |
| `GETTING_STARTED.md` | Setting up your computer |
| `ARCHITECTURE.md` | Why the code is organised this way |
| `ALGORITHM_TEMPLATE.md` | The exact shape a new algorithm should follow |
| `STYLE_GUIDE.md` | How to write code and explanations here |
| `TESTING.md` | How to write tests |
| `VISUALISATION.md` | How to add something to the visualiser |
| `ALGORITHMS.md` | The list of every algorithm and how finished it is |

## `mlblueprint/`

The actual library. Everything above exists to support what happens in here.

A folder with an `__init__.py` file inside it is what Python calls a "package", it just means
Python can import it. That's why `from mlblueprint.core import Estimator` works.

`core/` holds code every algorithm shares: input checking, random seeding, scoring, small
sample datasets. The other folders (`linear/`, `tree/`, `cluster/`, and so on) are where
algorithms live, grouped by type. They're empty right now, waiting for contributions.

## `apps/visualiser/`

One small app that shows algorithms working, step by step. Right now it's empty because there
are no algorithms yet, that's expected, not broken.

## `tests/`

Tests for the shared `core` code, and a few tests that check the project follows its own
rules (like making sure algorithm folders don't accidentally depend on each other).

## Things you'll see but never need to touch

| Name | What it is |
|---|---|
| `.venv/` | Your isolated Python setup for this project |
| `__pycache__/` | Compiled Python files, created automatically |
| `.pytest_cache/`, `.ruff_cache/` | Speed up repeat test and lint runs |
| `.git/` | Git's own storage of every past change |

Safe to delete any of these, they just get rebuilt.

## Adding maintainers

Once you've decided on your 3 or 4 maintainers:

1. **Add them to the repository.** On GitHub: Settings, Collaborators and teams, Add people.
   Search their GitHub username and give them "Write" access (or "Maintain" if you want them
   able to change repo settings too).
2. **Add their usernames to `MAINTAINERS.md`.** There's a table at the top with slots for
   this.
3. **Add their usernames to `.github/CODEOWNERS`**, so GitHub automatically asks them to
   review relevant pull requests. Replace the placeholder lines with your actual usernames,
   for example:

   ```
   *   @akshatidk29 @your-friend-handle
   ```

You don't need GitHub "teams" for 3 or 4 people, just list usernames directly, it's simpler
and does the same job at this size.

## Glossary

**Repository (repo)**, the whole project folder, with its full history of changes.

**Git vs GitHub**, git is the tool that tracks changes on your computer. GitHub is the website
that hosts the project online and adds things like issues and pull requests. Git works fine
without any internet connection.

**Commit**, a saved snapshot of some changes, with a short message describing them.

**Branch**, a separate line of work, so you can make changes without touching the main version
until you're ready. `main` is the primary branch.

**Push / pull**, uploading your commits to GitHub, or downloading someone else's.

**Fork**, your own personal copy of someone else's project on GitHub. How outside contributors
work on a project they can't directly edit.

**Pull request (PR)**, "please add my changes to your project." This is where review and
discussion happen before something gets merged in.

**Merge**, accepting a pull request's changes into `main`.

**Issue**, a tracked task, bug, or question. Most work starts as an issue.

**CI**, short for Continuous Integration, the automatic checks that run on every push.

**Linter / formatter**, tools that check and fix code style automatically. Ours is called
Ruff.

**Virtual environment**, an isolated Python setup just for this project, so it doesn't mix
with anything else on your computer.

**Package**, two meanings here: a folder Python can import, and a project installable with
`pip`. `mlblueprint` is both.

**Maintainer**, someone who can approve and merge pull requests, and is responsible for part
of the project.
