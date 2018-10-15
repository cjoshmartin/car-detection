# layers
import numpy
from numpy import uint16, sum
from numpy.core.multiarray import zeros, arange
from numpy.core.umath import floor, ceil
import matplotlib.pyplot as plt

from utils.general import dimensions, print_error
from utils.parse_images import save_image
from utils.ploting import graphs, save_plot


def conv_(img, conv_filter):
    filter_size = conv_filter.shape[1]
    result = numpy.zeros((img.shape))
    # Looping through the image to apply the convolution operation.
    for r in numpy.uint16(numpy.arange(filter_size / 2.0,
                                       img.shape[0] - filter_size / 2.0 + 1)):
        for c in numpy.uint16(numpy.arange(filter_size / 2.0,
                                           img.shape[1] - filter_size / 2.0 + 1)):
            """
            Getting the current region to get multiplied with the filter.
            How to loop through the image and get the region based on 
            the image and filer sizes is the most tricky part of convolution.
            """
            curr_region = img[r - numpy.uint16(numpy.floor(filter_size / 2.0)):r + numpy.uint16(
                numpy.ceil(filter_size / 2.0)),
                          c - numpy.uint16(numpy.floor(filter_size / 2.0)):c + numpy.uint16(
                              numpy.ceil(filter_size / 2.0))]
            # Element-wise multipliplication between the current region and the filter.
            curr_result = curr_region * conv_filter
            conv_sum = numpy.sum(curr_result)  # Summing the result of multiplication.
            result[r, c] = conv_sum  # Saving the summation in the convolution layer feature map.

    # Clipping the outliers of the result matrix.
    final_result = result[numpy.uint16(filter_size / 2.0):result.shape[0] - numpy.uint16(filter_size / 2.0),
                   numpy.uint16(filter_size / 2.0):result.shape[1] - numpy.uint16(filter_size / 2.0)]
    return final_result


def conv(img, conv_filter):
    if len(img.shape) > 2 or len(conv_filter.shape) > 3:  # Check if number of image channels matches the filter depth.
        if img.shape[-1] != conv_filter.shape[-1]:
            print("Error: Number of channels in both image and filter must match.")
            sys.exit()
    if conv_filter.shape[1] != conv_filter.shape[2]:  # Check if filter dimensions are equal.
        print('Error: Filter must be a square matrix. I.e. number of rows and columns must match.')
        sys.exit()
    if conv_filter.shape[1] % 2 == 0:  # Check if filter diemnsions are odd.
        print('Error: Filter must have an odd size. I.e. number of rows and columns must be odd.')
        sys.exit()

    # An empty feature map to hold the output of convolving the filter(s) with the image.
    feature_maps = numpy.zeros((img.shape[0] - conv_filter.shape[1] + 1,
                                img.shape[1] - conv_filter.shape[1] + 1,
                                conv_filter.shape[0]))

    # Convolving the image by the filter(s).
    for filter_num in range(conv_filter.shape[0]):
        print("Filter ", filter_num + 1)
        curr_filter = conv_filter[filter_num, :]  # getting a filter from the bank.
        """ 
        Checking if there are mutliple channels for the single filter.
        If so, then each channel will convolve the image.
        The result of all convolutions are summed to return a single feature map.
        """
        if len(curr_filter.shape) > 2:
            conv_map = conv_(img[:, :, 0], curr_filter[:, :, 0])  # Array holding the sum of all feature maps.
            for ch_num in range(1, curr_filter.shape[
                -1]):  # Convolving each channel with the image and summing the results.
                conv_map = conv_map + conv_(img[:, :, ch_num],
                                            curr_filter[:, :, ch_num])
        else:  # There is just a single channel in the filter.
            conv_map = conv_(img, curr_filter)
        feature_maps[:, :, filter_num] = conv_map  # Holding feature map with the current filter.
    return feature_maps  # Returning all feature maps.


def relu(feature_map):  # Activation function, normalize of what is passed from the convolution stage
    output = zeros(feature_map.shape)

    for map_num in range(feature_map.shape[-1]):
        for r in arange(0, feature_map.shape[0]):
            for c in arange(0, feature_map.shape[1]):
                output[r, c, map_num] = numpy.max([feature_map[r, c, map_num], 0])

    return output


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

    def activate(self):
        print('{} Activation'.format(self.layer_name))
        self.feature_maps = conv(self.__source, self.filter)
        self.activated_maps = self.__activation_func(self.feature_maps)
        self.reduced_maps = pooling(self.activated_maps)

    def plot_maps(self, rows, cols):
        fig, axes = plt.subplots(nrows=rows, ncols=cols)
        graphs(axes[0], self.feature_maps, '{}-Map{}', self.layer_name)
        graphs(axes[1], self.activated_maps, '{}-Map{}ReLU', self.layer_name)
        graphs(axes[2], self.reduced_maps, '{}-Map{}ReLUPool', self.layer_name)
        save_plot(fig, self.layer_name)
