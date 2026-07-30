# The visualiser

One app. Every algorithm can add a panel to it.

```bash
pip install -e ".[dev]"
streamlit run apps/visualiser/app.py
```

## How it works

1. The app looks through the `mlblueprint` package for any file named `viz.py`.
2. Each `viz.py` registers a panel using `@register_panel`.
3. The app lists whatever is registered, shows its settings as sidebar controls, and steps
   through the frames it returns.

So an algorithm just needs to add a `viz.py`, nothing else to wire up or register by hand.

## Writing a panel

See [docs/VISUALISATION.md](../../docs/VISUALISATION.md) for a full walkthrough.

## Why one app instead of one per algorithm

If every algorithm had its own separate app, we'd end up with dozens of small apps quietly
breaking over time, with nobody noticing since nothing checks them automatically. One shared
app is much easier to keep working, and panels return plain data, so they can be tested
without opening a browser at all.

## Ways to improve this app

It's basic right now, on purpose, just enough to work.

Open an issue first if you're planning something big here, since this app is shared by
everyone.
