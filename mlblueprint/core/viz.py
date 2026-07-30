"""The visualisation panel protocol and registry.

A panel does not draw to Streamlit directly. It produces a list of
:class:`Frame` objects, one per step of the algorithm, each of which knows how
to draw itself onto a matplotlib axes. That keeps panels testable as plain
Python objects, and keeps the rendering layer replaceable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Parameter",
    "Frame",
    "VizPanel",
    "register_panel",
    "get_panels",
    "discover_panels",
]


@dataclass(frozen=True)
class Parameter:
    """A control the user can adjust in the visualiser.

    Attributes
    ----------
    name : str
        Keyword argument name passed to :meth:`VizPanel.frames`.
    label : str
        Human-readable label shown next to the control.
    default : Any
        Starting value.
    kind : str
        One of ``"int"``, ``"float"``, ``"choice"`` or ``"bool"``.
    min, max, step : float or None
        Bounds for numeric controls.
    options : list or None
        Allowed values for ``"choice"`` controls.
    help : str
        One-line explanation shown as a tooltip. Worth writing, this is often
        where the teaching happens.
    """

    name: str
    label: str
    default: Any
    kind: str = "int"
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: list[Any] | None = None
    help: str = ""


@dataclass
class Frame:
    """One step of an algorithm, ready to be drawn.

    Attributes
    ----------
    draw : callable
        Called with a matplotlib ``Axes`` and draws this step onto it.
    caption : str
        What happened in this step. Shown under the plot.
    metrics : dict
        Optional numbers to display alongside, loss, inertia, iteration count.
    """

    draw: Callable[[Any], None]
    caption: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


class VizPanel(ABC):
    """Base class for an algorithm's visualisation.

    Subclasses set :attr:`name`, :attr:`family` and :attr:`description`, then
    implement :meth:`parameters` and :meth:`frames`.

    Examples
    --------
    ::

        from mlblueprint.core.viz import Frame, Parameter, VizPanel, register_panel

        @register_panel
        class KMeansPanel(VizPanel):
            name = "k-Means convergence"
            family = "cluster"
            description = "Watch centroids move until the assignments stop changing."

            def parameters(self):
                return [Parameter("k", "Clusters", 3, "int", min=2, max=8)]

            def frames(self, k=3):
                ...
                return [Frame(draw=..., caption="Iteration 1")]
    """

    #: Title shown in the visualiser.
    name: str = "Unnamed panel"

    #: Family folder this panel belongs to, e.g. ``"cluster"``.
    family: str = "misc"

    #: One sentence on what the panel shows.
    description: str = ""

    @abstractmethod
    def parameters(self) -> list[Parameter]:
        """Return the controls this panel exposes.

        Returns
        -------
        params : list of Parameter
            May be empty if the panel takes no configuration.
        """

    @abstractmethod
    def frames(self, **params: Any) -> list[Frame]:
        """Run the algorithm and return one frame per step.

        Parameters
        ----------
        **params
            Values for the controls declared by :meth:`parameters`, keyed by
            ``Parameter.name``.

        Returns
        -------
        frames : list of Frame
            Ordered from the first step to convergence. Keep the number modest,
            a few dozen frames is plenty, and the user has to step through them.
        """


_REGISTRY: dict[str, type[VizPanel]] = {}


def register_panel(panel_cls: type[VizPanel]) -> type[VizPanel]:
    """Register a panel class so the visualiser can find it.

    Use as a decorator on the panel class.

    Parameters
    ----------
    panel_cls : type[VizPanel]
        The panel to register.

    Returns
    -------
    panel_cls : type[VizPanel]
        The same class, unchanged, so the decorator is transparent.

    Raises
    ------
    TypeError
        If the class is not a :class:`VizPanel` subclass.
    """
    if not (isinstance(panel_cls, type) and issubclass(panel_cls, VizPanel)):
        raise TypeError(
            f"register_panel expects a VizPanel subclass, got {panel_cls!r}."
        )

    key = f"{panel_cls.family}.{panel_cls.__name__}"
    _REGISTRY[key] = panel_cls
    return panel_cls


def get_panels() -> dict[str, type[VizPanel]]:
    """Return every registered panel, keyed by ``family.ClassName``.

    Call :func:`discover_panels` first, panels only register when the module
    defining them has been imported.

    Returns
    -------
    panels : dict[str, type[VizPanel]]
        A copy of the registry.
    """
    return dict(_REGISTRY)


def discover_panels(package: str = "mlblueprint") -> dict[str, type[VizPanel]]:
    """Import every ``viz`` module under ``package`` so its panels register.

    Walks the package tree looking for modules named ``viz`` inside algorithm
    folders, and imports them. Import failures are skipped rather than raised,
    one algorithm with a broken optional dependency should not take the whole
    visualiser down with it.

    Parameters
    ----------
    package : str, default="mlblueprint"
        Package to search.

    Returns
    -------
    panels : dict[str, type[VizPanel]]
        Everything registered after the walk.
    """
    import importlib
    import pkgutil
    import warnings

    root = importlib.import_module(package)

    for module in pkgutil.walk_packages(root.__path__, prefix=f"{package}."):
        if not module.name.endswith(".viz"):
            continue
        # mlblueprint.core.viz is this module, not a panel.
        if module.name == f"{package}.core.viz":
            continue
        try:
            importlib.import_module(module.name)
        except Exception as exc:  # noqa: BLE001 - one bad panel must not break the app
            warnings.warn(
                f"Could not import visualisation panel {module.name}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    return get_panels()
