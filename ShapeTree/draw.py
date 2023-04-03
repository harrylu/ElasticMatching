import matplotlib.pyplot as plt

import plt_utils
if __name__ == "__main__":
    fig, ax = plt.subplots()
    ax.set_title('click to build line segments')
    line, = ax.plot([], [])  # empty line
    ax.set_xlim(-10,10); ax.set_ylim(-10,10)
    linebuilder = plt_utils.LineBuilder(line)
    plt.show()