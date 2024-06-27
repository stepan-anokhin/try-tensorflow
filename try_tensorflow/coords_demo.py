import os
from dataclasses import dataclass, field
from functools import partial
from typing import List, Tuple, TypeAlias, Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.backend_managers import ToolManager
from matplotlib.backend_tools import ToolBase, ToolToggleBase
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from numba import njit

matplotlib.rcParams["toolbar"] = "toolmanager"
t = np.arange(0.0, 1.0, 0.01)
s = np.sin(2 * np.pi * t)

fig: Figure = plt.figure()
fig.set_size_inches(12, 5)
axes = fig.subplots(1, 2)
params_axes: Axes = axes[0]
plot_axes: Axes = axes[1]

params_axes.set_xlim(-1, 1)
params_axes.set_ylim(-1, 1)
params_axes.set_title("Parameters")
params_axes.set_xlabel("a")
params_axes.set_ylabel("b")

plot_axes.set_xlim(-1, 1)
plot_axes.set_ylim(-1, 1)
plot_axes.set_title("y = a * x + b")
plot_axes.set_xlabel("x")
plot_axes.set_ylabel("y")


def real_y(x):
    return 0.45 * x + 0.3


x = np.linspace(-1, 1, 50)
noise = np.random.normal(0, 0.2, len(x))
true_y = real_y(x)
y = real_y(x) + noise

plot_axes.scatter(x, y, color="black")


@njit()
def calc_error(x, y, a, b):
    predicted_y = x * a - b
    return np.sqrt(sum((predicted_y - y) ** 2) / len(x))


bound_calc_error = partial(calc_error, x, y)

steps = 10
errors = np.array([[calc_error(x, y, a, b) for a in np.linspace(-1, 1, steps)] for b in np.linspace(-1, 1, steps)])
for b in np.linspace(-1, 1, steps):
    for a in np.linspace(-1, 1, steps):
        da = db = 0.1
        df_da = (bound_calc_error(a + da, -b) - bound_calc_error(a, -b)) / da
        df_db = (bound_calc_error(a, -b + db) - bound_calc_error(a, -b)) / db
        params_axes.arrow(a, b, - 0.1 * df_da, 0.1 * df_db, head_width=0.03)

image = params_axes.imshow(errors, extent=(-1, 1, -1, 1), interpolation="bicubic")
colorbar: Colorbar = fig.colorbar(image, ax=params_axes, label="Error")
colorbar_axes: Axes = colorbar.ax


@dataclass
class State:
    figure: Figure
    params_axes: Axes
    plot_axes: Axes
    colorbar_axes: Axes
    pressed: bool = False
    a: float = None
    b: float = None
    interactive: bool = True
    movable: List[Artist] = field(default_factory=list)
    history: List[Tuple[float, float]] = field(default_factory=list)

    def clean(self):
        for artist in self.movable:
            artist.remove()
        self.movable.clear()

    def update(self, a: float, b: float):
        self.clean()
        self.a = a
        self.b = b

        self.movable.append(self.params_axes.scatter([a], [b], color="white"))
        self.movable.append(self.params_axes.axhline(y=b, xmin=-1, xmax=1, linestyle="--", color="white"))
        self.movable.append(self.params_axes.axvline(x=a, ymin=-1, ymax=1, linestyle="--", color="white"))

        params_text = f"a={a:0.2f}, b={b:0.2f}"
        self.movable.append(self.params_axes.text(a + 0.1, b + 0.1, params_text))

        current_error = bound_calc_error(a, -b)
        self.movable.append(self.colorbar_axes.scatter([0.5], current_error, color="white"))

        fit_x = np.linspace(-1, 1, 2)
        fit_y = fit_x * a + b
        self.movable.extend(self.plot_axes.plot(fit_x, fit_y, color="blue"))

        predicted_y = x * a + b
        for i in range(len(x)):
            error_line = self.plot_axes.axvline(
                x=x[i], ymin=min(y[i], predicted_y[i]) / 2 + 0.5,
                ymax=max(y[i], predicted_y[i]) / 2 + 0.5, color="red",
                linestyle="--")
            self.movable.append(error_line)
        fig.canvas.draw()

    def step(self):
        if self.a is None or self.b is None:
            return
        a, b = self.a, self.b
        da = db = 0.1
        df_da = (bound_calc_error(a + da, -b) - bound_calc_error(a, -b))
        df_db = (bound_calc_error(a, -b + db) - bound_calc_error(a, -b))
        self.update(a - 2 * df_da, b + 2 * df_db)
        self.history.append((a, b))
        self.draw_history()

    def draw_history(self):
        for a, b in self.history:
            self.movable.append(self.params_axes.scatter([a], [b], color="white"))
        for i in range(1, len(self.history)):
            a1, b1 = self.history[i - 1]
            a2, b2 = self.history[i]
            self.movable.extend(self.params_axes.plot([a1, a2], [b1, b2], color="white"))

        a1, b1 = self.history[-1]
        a2, b2 = self.a, self.b
        self.movable.extend(self.params_axes.plot([a1, a2], [b1, b2], color="white"))
        self.figure.canvas.draw()

    def clear_history(self):
        self.history.clear()


state = State(fig, params_axes, plot_axes, colorbar_axes)


def on_move(event):
    if event.inaxes == params_axes and state.pressed and state.interactive:
        state.update(event.xdata, event.ydata)


def on_press(event):
    if event.inaxes == params_axes:
        state.pressed = True
        if state.interactive:
            state.update(event.xdata, event.ydata)
            state.clear_history()


def on_release(event):
    state.pressed = False


fig.canvas.mpl_connect('motion_notify_event', on_move)
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)


class MakeStep(ToolBase):
    image = os.path.join(os.getcwd(), "next.png")

    def trigger(self, sender, event, data=None):
        state.step()


class ToggleInteractive(ToolToggleBase):
    image = os.path.join(os.getcwd(), "interactive.png")

    def enable(self, event=None):
        state.interactive = True

    def disable(self, event=None):
        state.interactive = False


print(os.getcwd())
tools: ToolManager = fig.canvas.manager.toolmanager
tools.add_tool("make_step", MakeStep)
fig.canvas.manager.toolbar.add_tool(tools.get_tool("make_step"), "toolgroup")
tools.add_tool("toggle_interactive", ToggleInteractive)
fig.canvas.manager.toolbar.add_tool(tools.get_tool("toggle_interactive"), "toolgroup")
plt.show()
