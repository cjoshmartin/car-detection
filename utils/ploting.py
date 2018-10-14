import os

from matplotlib import pyplot as plt


def graphs(axes, maps, text_to, layer_):
    for j in range(0, maps.shape[2]):
        plot(
            axes[j],
            maps[:, :, j],
            text_to.format(layer_, j + 1)
        )


def plot(axes, arr_to_write, title):
    axes.imshow(arr_to_write).set_cmap('gray')
    axes.get_xaxis().set_ticks([])
    axes.get_yaxis().set_ticks([])
    axes.set_title(title)


def save_plot(fig, location):
    if not os.path.isdir('./output'):
        os.makedirs('./output')

    plt.savefig('output/{}.png'.format(location), bbox_inches="tight")
    plt.close(fig)
