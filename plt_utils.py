import matplotlib
import matplotlib.pyplot as plt
import math

# xs, ys are 1 dimensional arrays
# xlim, ylim are 2-tuples (min, max)
def plot_coordinates(xs, ys, xlim, ylim, fig = None, **scatter_kwargs):
    fig = plt.figure() if fig is None else fig
    ax = plt.gca()
    ax.scatter(xs, ys, **scatter_kwargs)
    ax.set(xlim=xlim, ylim=ylim, aspect='equal')

    # Set bottom and left spines as x and y axes of coordinate system
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xticks(plt.xticks()[0][plt.xticks()[0] != 0])
    ax.set_yticks(plt.yticks()[0][plt.yticks()[0] != 0])

    ax.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)
    return fig

class LineBuilder:
    # calls callback with figure, event after adding point but before canvas draw
    def __init__(self, line, precision=0.1, callback=None):
        self.line = line
        self.precision = precision
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())

        self.motion_notify_cid = line.figure.canvas.mpl_connect('motion_notify_event', self)
        self.callback = callback

    def __call__(self, event):
        # print('click', event)
        if event.button != 1 or event.inaxes != self.line.axes: return
        xydata = self.line.get_xydata()
        if len(xydata) == 0:
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
            self.line.set_data(self.xs, self.ys)

        if math.dist(self.line.get_xydata()[-1], (event.xdata, event.ydata)) >= self.precision:
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
            self.line.set_data(self.xs, self.ys)
            if self.callback is not None:
                self.callback(self.line, event)
            self.line.figure.canvas.draw()

if __name__ == "__main__":
    pass