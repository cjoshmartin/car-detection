# layers
import numpy
from numpy import uint16
from numpy.core.multiarray import zeros, arange
import matplotlib.pyplot as plt

from neuron.convolution import conv
from utils.general import dimensions
from utils.parse_images import save_image
from utils.ploting import graphs, save_plot


def max_pooling(feature_map, size=2, stride=2):
    # Preparing the output of the pooling operation.
    # https://www.quora.com/What-is-max-pooling-in-convolutional-neural-networks

    d_out = dimensions(feature_map.shape, size)
    pool_out = zeros((uint16(d_out[0] / stride + 1),
                      uint16(d_out[1] / stride + 1),
                      feature_map.shape[-1]))

    for map_num in range(feature_map.shape[-1]):
        row2 = 0
        for row in arange(0, d_out[0], stride):
            column2 = 0
            for column in arange(0, d_out[1], stride):
                pool_out[row2, column2, map_num] = numpy.max([feature_map[row : row + size, column : column + size]]) # finds the max out of range of values
                column2 = column2 + 1
            row2 = row2 + 1
    return pool_out


# Neuron
class Neuron:
    def __init__(self, source, activation, layer_name, filter=None):

        self.layer_name = layer_name
        self.__source = source
        self.__activation_func = activation

        self.feature_maps = None
        self.activated_maps = None
        self.reduced_maps = None
        self.shouldPool = True

        if filter is None:
            # input layer
            save_image('Input', source)

            self.filter = numpy.zeros((2, 3, 3))
            self.filter[0, :, :] = numpy.array([[[-1, 0, 1],
                                                 [-1, 0, 1],     # Vertical Lines
                                                 [-1, 0, 1]]])
            self.filter[1, :, :] = numpy.array([[[1, 1, 1],
                                                 [0, 0, 0],      # Horizontal Lines
                                                 [-1, -1, -1]]])
        else:
            self.filter = filter

    def set_image(self, image):
        self.__source = image

    def set_should_pooling(self, val):
        self.shouldPool = val

    def get_map(self):
        if self.reduced_maps is not None:
            return self.reduced_maps
        if self.activated_maps is not None:
            return self.activated_maps
        if self.feature_maps is not None:
            return self.feature_maps

        print('Man you messed something up, look at your code for your maps')

    def toggle_pooling(self):
        self.shouldPool = not self.shouldPool

    def activate(self):
        print('{} Activation'.format(self.layer_name))
        self.feature_maps = conv(self.__source, self.filter)
        self.activated_maps = self.__activation_func(self.feature_maps)

        if self.shouldPool:
            self.reduced_maps = max_pooling(self.activated_maps)

    def plot_maps(self, i):
        fig, axes = plt.subplots(nrows=3, ncols=3)
        graphs(axes[0], self.feature_maps, '{}-Map{}', self.layer_name)
        graphs(axes[1], self.activated_maps, '{}-Map{}ReLU', self.layer_name)
        if self.shouldPool:
            graphs(axes[2], self.reduced_maps, '{}-Map{}ReLUPool', self.layer_name)

        save_plot(fig, '{}-{}'.format(i, self.layer_name))
