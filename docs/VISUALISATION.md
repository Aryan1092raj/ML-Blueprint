# Adding a visualisation

Watching an algorithm run is often the fastest way to understand it. Gradient descent rolling
downhill, k-means centroids settling into place, that kind of thing.

There is one visualiser app for the whole project. You add a small file, and your algorithm
shows up in it automatically, no extra setup needed.

```bash
streamlit run apps/visualiser/app.py
```

## How it works

Your panel does not draw anything directly. It just returns a list of steps ("frames"), and
each one knows how to draw itself. This keeps things simple to test, and means we can change
how things are displayed later without touching every panel.

## Writing one

Add `viz.py` to your algorithm's folder:

```python
from mlblueprint.core.datasets import make_blobs
from mlblueprint.core.viz import Frame, Parameter, VizPanel, register_panel


@register_panel
class KMeansPanel(VizPanel):
    name = "k-Means convergence"
    family = "cluster"
    description = "Watch the centroids move until they stop changing."

    def parameters(self):
        return [
            Parameter("k", "Clusters", default=3, kind="int", min=2, max=8),
        ]

    def frames(self, k=3):
        X, _ = make_blobs(n_samples=300, centers=k, random_state=42)

        # run the algorithm, saving the state at each step
        frames = []
        for step, (centers, labels) in enumerate(history):

            def draw(ax, centers=centers, labels=labels):
                ax.scatter(X[:, 0], X[:, 1], c=labels)
                ax.scatter(
                    centers[:, 0], centers[:, 1], marker="x", s=200, color="black"
                )

            frames.append(Frame(draw=draw, caption=f"Step {step + 1}"))
        return frames
```

That's the whole thing. No list to update anywhere, the app finds it on its own.

**One thing to watch for:** in the example above, `centers=centers, labels=labels` inside
`def draw(ax, ...)` is needed. Without it, every frame would show the very last step instead
of its own step, that's just how Python closures work.

## A few tips

- **Fix the random seed**, so the panel looks the same every time someone runs it.
- **Keep it fast**, a second or two total. Small datasets (a couple hundred points) are usually
  easier to read anyway.
- **Show what's actually happening.** Captions and small numbers (like "loss: 0.4") help more
  than a pretty picture on its own.
- **Let people break it.** Letting someone set `k` wrong, or crank up the learning rate until
  it diverges, teaches more than a perfect run does.

## Testing a panel

No browser needed, since it just returns data:

```python
def test_panel_produces_frames():
    panel = KMeansPanel()
    frames = panel.frames(k=3)
    assert len(frames) > 1
```