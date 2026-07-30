"""The ML Blueprint visualiser.

One application, many panels. Algorithms contribute a ``viz.py`` that
registers a ``VizPanel``, and this app discovers and renders them.

Run it with::

    streamlit run apps/visualiser/app.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from mlblueprint.core.viz import Parameter, VizPanel, discover_panels


def render_control(param: Parameter):
    """Render one panel parameter as a Streamlit widget.

    Parameters
    ----------
    param : Parameter
        The control to render.

    Returns
    -------
    value : Any
        The value the user selected.
    """
    if param.kind == "choice":
        return st.sidebar.selectbox(
            param.label, param.options or [], help=param.help or None
        )

    if param.kind == "bool":
        return st.sidebar.checkbox(
            param.label, value=bool(param.default), help=param.help or None
        )

    if param.kind == "float":
        return st.sidebar.slider(
            param.label,
            min_value=float(param.min if param.min is not None else 0.0),
            max_value=float(param.max if param.max is not None else 1.0),
            value=float(param.default),
            step=float(param.step) if param.step else None,
            help=param.help or None,
        )

    return st.sidebar.slider(
        param.label,
        min_value=int(param.min if param.min is not None else 0),
        max_value=int(param.max if param.max is not None else 100),
        value=int(param.default),
        step=int(param.step) if param.step else 1,
        help=param.help or None,
    )


def main() -> None:
    """Run the visualiser."""
    st.set_page_config(page_title="ML Blueprint Visualiser", layout="wide")

    st.title("ML Blueprint Visualiser")
    st.caption(
        "Algorithms, one step at a time. Every panel here is contributed by an "
        "algorithm folder in the library."
    )

    panels = discover_panels()

    if not panels:
        st.info(
            "**No visualisation panels yet.**\n\n"
            "Panels come from algorithm folders. An algorithm adds a `viz.py` "
            "that registers a `VizPanel`, and it shows up here automatically.\n\n"
        )
        return

    st.sidebar.header("Panel")
    labels = {f"{cls.family} · {cls.name}": key for key, cls in sorted(panels.items())}
    chosen = st.sidebar.selectbox("Algorithm", sorted(labels))
    panel_cls = panels[labels[chosen]]
    panel: VizPanel = panel_cls()

    if panel.description:
        st.subheader(panel.name)
        st.write(panel.description)

    st.sidebar.header("Parameters")
    values = {p.name: render_control(p) for p in panel.parameters()}

    try:
        frames = panel.frames(**values)
    except Exception as exc:
        st.error(f"This panel raised an error while running:\n\n`{exc}`")
        return

    if not frames:
        st.warning("This panel produced no frames.")
        return

    st.sidebar.header("Playback")
    index = st.sidebar.slider("Step", 0, len(frames) - 1, 0) if len(frames) > 1 else 0
    frame = frames[index]

    plot_column, info_column = st.columns([3, 1])

    with plot_column:
        fig, ax = plt.subplots(figsize=(7, 5))
        frame.draw(ax)
        st.pyplot(fig)
        plt.close(fig)
        if frame.caption:
            st.caption(frame.caption)

    with info_column:
        st.metric("Step", f"{index + 1} / {len(frames)}")
        for label, value in frame.metrics.items():
            st.metric(label, f"{value:.4g}" if isinstance(value, float) else value)


if __name__ == "__main__":
    main()
