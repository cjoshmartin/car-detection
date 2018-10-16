# layers
import numpy
from numpy import uint16, sum
from numpy.core.multiarray import zeros, arange
from numpy.core.umath import floor, ceil
import matplotlib.pyplot as plt

from utils.general import dimensions, print_error
from utils.parse_images import save_image
from utils.ploting import graphs, save_plot

def dot_product(image, filter):
    # Element-wise multiplication between the current region and the filter.
    element_wise_multip = image * filter
    return numpy.sum(element_wise_multip) # summing the results of the multiplication

def convolution_range(half_filter_size,location, stride=1):
    __range = numpy.arange(half_filter_size, location - half_filter_size + 1, stride)
    return numpy.uint16(__range)

def conv_(img, conv_filter):
    filter_size = conv_filter.shape[1]
    half_filter_size = filter_size / 2.0
    result = numpy.zeros(img.shape)


    row_range = convolution_range(half_filter_size, img.shape[0])
    column_range = convolution_range(half_filter_size, img.shape[1])

    # Looping through the image to apply the convolution operation.
    for row in row_range:
        for column in column_range:
            """
            Getting the current region to get multiplied with the filter.
            How to loop through the image and get the region based on 
            the image and filer sizes is the most tricky part of convolution.
            """
            floor_of_filter = numpy.uint16(numpy.floor(half_filter_size))
            ceil_of_filter = numpy.uint16(numpy.ceil(half_filter_size))

            curr_region = img[
                          row - floor_of_filter : row + ceil_of_filter,
                          column - floor_of_filter : column + ceil_of_filter
                          ]

            result[row, column] = dot_product(curr_region, conv_filter)  # Saving the summation in the convolution layer feature map.

    # Clipping the outliers of the result matrix.
    hfs_int = numpy.uint16(half_filter_size) # converting half_filter_size to an int
    return result[
                   hfs_int: result.shape[0] - hfs_int,
                   hfs_int: result.shape[1] - hfs_int
                   ]


def conv(img, conv_filter):
    if len(img.shape) > 2 or len(conv_filter.shape) > 3:  # Check if number of image channels matches the filter depth.
        if img.shape[-1] != conv_filter.shape[-1]:
            print_error("Error: Number of channels in both image and filter must match.")
    if conv_filter.shape[1] != conv_filter.shape[2]:  # Check if filter dimensions are equal.
        print_error('Error: Filter must be a square matrix. I.e. number of rows and columns must match.')
    if conv_filter.shape[1] % 2 == 0:  # Check if filter diemnsions are odd.
        print_error('Error: Filter must have an odd size. I.e. number of rows and columns must be odd.')

    # An empty feature map to hold the output of convolving the filter(s) with the image.
    d_out = dimensions(img.shape, conv_filter.shape[1], conv_filter.shape[0])
    feature_maps = numpy.zeros(d_out)

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
            for ch_num in range(1, curr_filter.shape[-1]):  # Convolving each channel with the image and summing the results.
                conv_map = conv_map + conv_(img[:, :, ch_num], curr_filter[:, :, ch_num])
        else:  # There is just a single channel in the filter.
            conv_map = conv_(img, curr_filter)
        feature_maps[:, :, filter_num] = conv_map  # Holding feature map with the current filter.
    return feature_maps  # Returning all feature maps.


def relu(feature_map):  # Activation function, normalize of what is passed from the convolution stage
    output = zeros(feature_map.shape)

    for map_num in range(feature_map.shape[-1]):
        for row in arange(0, feature_map.shape[0]):
            for column in arange(0, feature_map.shape[1]):
                output[row, column, map_num] = numpy.max([0, feature_map[row, column, map_num]])

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

    def set_image(self, image):
        self.__source = image

    def activate(self, shouldPool=True):
        print('{} Activation'.format(self.layer_name))
        self.feature_maps = conv(self.__source, self.filter)
        self.activated_maps = self.__activation_func(self.feature_maps)

        if shouldPool:
            self.reduced_maps = max_pooling(self.activated_maps)

    def plot_maps(self, i):
        fig, axes = plt.subplots(nrows=3, ncols=3)
        graphs(axes[0], self.feature_maps, '{}-Map{}', self.layer_name)
        graphs(axes[1], self.activated_maps, '{}-Map{}ReLU', self.layer_name)
        graphs(axes[2], self.reduced_maps, '{}-Map{}ReLUPool', self.layer_name)

        save_plot(fig, '{}-{}'.format(i, self.layer_name))
