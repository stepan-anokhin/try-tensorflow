import os.path
from dataclasses import dataclass
from typing import Tuple, TypeAlias, Callable, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.backend_managers import ToolManager
from matplotlib.backend_tools import ToolBase, ToolToggleBase
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

matplotlib.rcParams["toolbar"] = "toolmanager"

# Type alias for model function
ModelFunc: TypeAlias = Callable[[np.ndarray[float], float, float], np.ndarray[float]]


class ModelVisualizer2D:
    """Error visualizer for models with 2 parameters fitting functions with 1 parameter.

    The empiric data is specified as x and y arrays. The model is specified as a function
    taking 4 arguments: x, y, a and b, where a and b are the free parameters of the model.
    """

    @dataclass(frozen=True)
    class Config:
        x_range: Tuple[float, float] = None
        y_range: Tuple[float, float] = None
        a_range: Tuple[float, float] = (-1, 1)
        b_range: Tuple[float, float] = (-1, 1)
        fig_size: Tuple[float, float] = (12, 5)
        title: str = "Model fitting"
        title_params = "Parameters"
        title_plot = "Data"
        data_color = "black"
        model_color = "blue"
        error_bars_color = "red"
        params_color = "white"
        arrows_color = "blue"
        arrow_head_size: float = 0.03
        arrow_scale: float = 0.2
        approx_steps: int = 20
        da: float = 0.1
        db: float = 0.1
        descend_speed: float = 2.0
        errors_cmap: Colormap = cm.coolwarm

    def __init__(
            self,
            x: np.ndarray[float],
            y: np.ndarray[float],
            model: ModelFunc,
            config: Config = Config(),
    ):
        self.x: np.ndarray[float] = x
        self.y: np.ndarray[float] = y
        self.model: ModelFunc = model
        self.config: ModelVisualizer2D.Config = config
        self.model_x = np.linspace(self.x_min, self.x_max, self.steps)

        self.fig: Figure = plt.figure()
        self.fig.set_size_inches(*self.config.fig_size)

        self.params_axes: Axes = self.setup_params_axes(self.fig.add_subplot(2, 2, 1))
        self.plot_axes: Axes = self.setup_plot_axes(self.fig.add_subplot(1, 2, 2))
        self.errors_axes: Axes3D = self.setup_errors_axes(
            self.fig.add_subplot(2, 2, 3, projection='3d', computed_zorder=False))

        self.model_components: List[Artist] = []
        self.draw_data()

        self.colorbar: Colorbar = self.draw_error_colormap()
        self.draw_error_gradient()

        self.interactive: bool = True
        self.pressed: bool = False
        self.a: Optional[float] = None
        self.b: Optional[float] = None
        self.history: List[Tuple[float, float]] = []
        self.bind_events()
        self.bind_tools()
        self.fig.tight_layout(pad=0.2)

    @property
    def x_range(self) -> Tuple[float, float]:
        return self.config.x_range or (np.min(self.x), np.max(self.x))

    @property
    def y_range(self) -> Tuple[float, float]:
        return self.config.y_range or (np.min(self.y), np.max(self.y))

    @property
    def x_min(self) -> float:
        return self.x_range[0]

    @property
    def x_max(self) -> float:
        return self.x_range[1]

    @property
    def y_min(self) -> float:
        return self.y_range[0]

    @property
    def y_max(self) -> float:
        return self.y_range[1]

    @property
    def a_min(self) -> float:
        return self.config.a_range[0]

    @property
    def a_max(self) -> float:
        return self.config.a_range[1]

    @property
    def b_min(self) -> float:
        return self.config.b_range[0]

    @property
    def b_max(self) -> float:
        return self.config.b_range[1]

    @property
    def steps(self) -> int:
        return self.config.approx_steps

    def setup_params_axes(self, params_axes: Axes) -> Axes:
        """Draw basic elements of parameters axes."""
        params_axes.set_xlim(*self.config.a_range)
        params_axes.set_ylim(*self.config.b_range)
        params_axes.set_title(self.config.title_params)
        params_axes.set_xlabel("a")
        params_axes.set_ylabel("b")
        return params_axes

    def setup_plot_axes(self, plot_axes: Axes) -> Axes:
        """Draw basic elements of plot axes."""
        plot_axes.set_xlim(*self.config.x_range or (np.min(self.x), np.max(self.x)))
        plot_axes.set_ylim(*self.config.y_range or (np.min(self.y), np.max(self.y)))
        plot_axes.set_title(self.config.title_plot)
        plot_axes.set_xlabel("x")
        plot_axes.set_ylabel("y")
        return plot_axes

    def setup_errors_axes(self, errors_axes: Axes3D) -> Axes3D:
        """Draw 3D errors axes."""
        errors_axes.set_aspect('equal')
        errors_axes.set_title("Errors")
        errors_axes.set_xlim(*self.config.a_range)
        errors_axes.set_ylim(*self.config.b_range)
        errors_axes.set_xlabel("a")
        errors_axes.set_ylabel("b")
        errors_axes.set_zlabel("error")

        a_step = (self.a_max - self.a_min) / self.steps
        b_step = (self.b_max - self.b_min) / self.steps
        a_values, b_values = np.mgrid[self.a_min:self.a_max:a_step, self.b_min:self.b_max:b_step]
        calc_errors = np.vectorize(self.error)
        error_values = calc_errors(a_values, b_values)
        errors_axes.plot_surface(a_values, b_values, error_values, cmap=self.config.errors_cmap, alpha=0.8)

        z_min = np.min(error_values)
        z_max = np.max(error_values)
        z_span = (z_max - z_min)
        z_margin = 0.1 * z_span
        errors_axes.set_zlim(z_min - z_margin, z_max + z_margin)
        return errors_axes

    def draw_data(self):
        """Draw data points."""
        self.plot_axes.scatter(self.x, self.y, color=self.config.data_color)

    def error(self, a: float, b: float) -> float:
        """Calculate error for the given params."""
        predicted = self.model(self.x, a, b)
        return np.sqrt(sum((predicted - self.y) ** 2) / len(self.x))

    def error_map(self) -> np.ndarray:
        """Generate error map as 2D array."""
        a_values = np.linspace(self.a_min, self.a_max, self.steps)
        b_values = np.linspace(self.b_min, self.b_max, self.steps)
        return np.array([[self.error(a, -b) for a in a_values] for b in b_values])

    def draw_error_colormap(self) -> Colorbar:
        """Draw error map with colorbar."""
        errors = self.error_map()
        bounding_box = (self.a_min, self.a_max, self.b_min, self.b_max)
        image = self.params_axes.imshow(errors, extent=bounding_box, interpolation="bicubic",
                                        cmap=self.config.errors_cmap)
        return self.fig.colorbar(image, ax=self.params_axes, label="Error", cmap=self.config.errors_cmap)

    def draw_error_gradient(self):
        """Draw negative error gradient."""
        for b in np.linspace(self.b_min, self.b_max, int(self.steps / 2)):
            for a in np.linspace(self.a_min, self.a_max, int(self.steps / 2)):
                arrow = self.config.arrow_scale * self.gradient(a, b)
                self.params_axes.arrow(a, b, arrow[0], arrow[1], head_width=self.config.arrow_head_size)

    def gradient(self, a: float, b: float) -> np.ndarray:
        """Calculate error gradient in the point."""
        df_da = (self.error(a + self.config.da, b) - self.error(a, b)) / self.config.da
        df_db = (self.error(a, b + self.config.db) - self.error(a, b)) / self.config.db
        return -np.array([df_da, df_db])

    def draw_model(self, a: float, b: float):
        """Visualize model."""
        self.a = a
        self.b = b

        params_color = self.config.params_color
        self.model_components.append(self.params_axes.scatter([a], [b], color=params_color))
        self.model_components.append(self.params_axes.axhline(y=b, xmin=-1, xmax=1, linestyle="--", color=params_color))
        self.model_components.append(self.params_axes.axvline(x=a, ymin=-1, ymax=1, linestyle="--", color=params_color))

        params_text = f"a={a:0.2f}, b={b:0.2f}"
        self.model_components.append(self.params_axes.text(a + 0.1, b + 0.1, params_text))

        current_error = self.error(a, b)
        self.model_components.append(self.colorbar.ax.scatter([0.5], current_error, color=params_color))

        model_y = self.model(self.model_x, a, b)
        self.model_components.extend(self.plot_axes.plot(self.model_x, model_y, color=self.config.model_color))

        predicted = self.model(self.x, a, b)
        y_span = self.y_max - self.y_min
        for i in range(len(self.x)):
            y_min = max(self.y_min, min(predicted[i], self.y[i]))
            y_max = min(self.y_max, max(predicted[i], self.y[i]))

            error_line = self.plot_axes.axvline(
                x=self.x[i],
                ymin=(y_min - self.y_min) / y_span,
                ymax=(y_max - self.y_min) / y_span,
                color=self.config.error_bars_color,
                linestyle="--"
            )

            self.model_components.append(error_line)

        self.model_components.append(self.errors_axes.scatter([a], [b], [current_error], color=params_color))
        self.fig.canvas.draw()

    def clear_model(self):
        """Delete movable elements."""
        for artist in self.model_components:
            artist.remove()
        self.model_components.clear()

    def make_step(self):
        """Make step toward best fit."""
        if self.a is None or self.b is None:
            return

        step = self.config.descend_speed * self.gradient(self.a, self.b)
        self.history.append((self.a, self.b))
        self.clear_model()
        self.draw_model(self.a + step[0], self.b + step[1])
        self.draw_history()

    def draw_history(self):
        for a, b in self.history:
            self.model_components.append(self.params_axes.scatter([a], [b], color=self.config.params_color))
        for i in range(1, len(self.history)):
            a1, b1 = self.history[i - 1]
            a2, b2 = self.history[i]
            self.model_components.extend(self.params_axes.plot([a1, a2], [b1, b2], color=self.config.params_color))

        a1, b1 = self.history[-1]
        a2, b2 = self.a, self.b
        self.model_components.extend(self.params_axes.plot([a1, a2], [b1, b2], color=self.config.params_color))
        self.fig.canvas.draw()

    def clear_history(self):
        self.history.clear()

    def on_move(self, event):
        """Handle mouse move."""
        if event.inaxes == self.params_axes and self.pressed and self.interactive:
            self.clear_model()
            self.draw_model(a=event.xdata, b=event.ydata)

    def on_press(self, event):
        """Handle mouse button press."""
        if event.inaxes == self.params_axes:
            self.pressed = True
            if self.interactive:
                self.clear_history()
                self.clear_model()
                self.draw_model(a=event.xdata, b=event.ydata)

    def on_release(self, event):
        """Handle mose button release."""
        self.pressed = False

    def bind_events(self):
        """Bind mouse events."""
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)

    def bind_tools(self):
        visualizer: ModelVisualizer2D = self

        class MakeStep(ToolBase):
            """Make step tool."""
            image = os.path.join(os.path.dirname(__file__), "next.png")

            def trigger(self, sender, event, data=None):
                visualizer.make_step()

        class ToggleInteractive(ToolToggleBase):
            """Toggle interactive mode tool."""
            image = os.path.join(os.path.dirname(__file__), "interactive.png")

            def enable(self, event=None):
                visualizer.interactive = True

            def disable(self, event=None):
                visualizer.interactive = False

        tools: ToolManager = self.fig.canvas.manager.toolmanager
        tools.add_tool("make_step", MakeStep)
        self.fig.canvas.manager.toolbar.add_tool(tools.get_tool("make_step"), "toolgroup")
        tools.add_tool("toggle_interactive", ToggleInteractive)
        self.fig.canvas.manager.toolbar.add_tool(tools.get_tool("toggle_interactive"), "toolgroup")


def main():
    x = np.linspace(-1, 1, 50)
    y = np.sin(x) + np.random.normal(0, 0.2, len(x))

    def model0(x, a, b):
        return a * np.sin(x + b)

    def model1(x, a, b):
        return a * np.sin(x * b)

    def model2(x, a, b):
        return a * x + b

    def model3(x, a, b):
        return a * (x ** 2) + b * x + model0(x, a, b) + model1(x + 1, a, b)

    viz = ModelVisualizer2D(x, y, model3, config=ModelVisualizer2D.Config(
        a_range=(-4, 4),
        b_range=(-4, 4),
        y_range=(-4, 4),
        descend_speed=0.2
    ))
    plt.show()


if __name__ == '__main__':
    main()
