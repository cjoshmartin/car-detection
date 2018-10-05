# layers
import numpy
from numpy import uint16, sum
from numpy.core.multiarray import zeros, arange
from numpy.core.umath import floor, ceil
import matplotlib.pyplot as plt

from utils.general import demations, print_error
from utils.parse_images import save_image
from utils.ploting import graphs, save_plot


class Convolution:
    def __init__(self, img, conv_filter):
        if len(img.shape) > 2 or len(conv_filter.shape) > 3:
            if img.shape[-1] != conv_filter.shape[-1]:
                print_error('Error: Number of channels in both image and filter must match.')
            if conv_filter.shape[1] != conv_filter.shape[2]:
                print_error('Error: Filter must be a square matrix. I.e. number of rows and columns must match.')
            if conv_filter.shape[2] % 2 == 0:
                print_error('Error: Filter must have an odd size. I.e. number of rows and columns must be odd.')

        number_of_filters = conv_filter.shape[0]
        self.feature_maps = zeros(demations(img.shape, conv_filter.shape, number_of_filters))

        for filter_num in range(number_of_filters):
            print("Filter ", filter_num + 1)
            curr_filter = conv_filter[filter_num, :]  # getting a filter from the bank.

            if len(curr_filter.shape) > 2:
                conv_map = self.__conv(img[:, :, 0], curr_filter[:, :, 0])
                for channel_number in range(1, curr_filter.shape[
                    -1]):  # Convolving each channel with the image and summing the results.
                    conv_map = conv_map + self.__conv(img[:, :, channel_number], curr_filter[:, :, channel_number])
            else:
                conv_map = self.__conv(img, curr_filter)  # single channel in filter
                self.feature_maps[:, :, filter_num] = conv_map

    def __conv(self, img, conv_filter):
        filter_size = conv_filter.shape[1]
        result = zeros((img.shape))
        # Looping through the image to apply the convolution operation.
        size___ = filter_size / 2.0
        for r in uint16(arange(size___,
                               img.shape[0] - size___ + 1)):
            for c in uint16(arange(size___,
                                   img.shape[1] - size___ + 1)):
                """
                Getting the current region to get multiplied with the filter.
                How to loop through the image and get the region based on 
                the image and filer sizes is the most tricky part of convolution.
                """
                curr_region = img[r - uint16(floor(size___)):r + uint16(
                    ceil(size___)),
                              c - uint16(floor(size___)): c + uint16(
                                  ceil(size___))]
                # Element-wise multipliplication between the current region and the filter.
                curr_result = curr_region * conv_filter
                conv_sum = sum(curr_result)  # Summing the result of multiplication.
                result[r, c] = conv_sum  # Saving the summation in the convolution layer feature map.

        # Clipping the outliers of the result matrix.
        final_result = result[uint16(size___): result.shape[0] - uint16(size___),
                       uint16(size___): result.shape[1] - uint16(size___)]
        return final_result


def relu(feature_map, activation):  # Activation function, normalize of what is passed from the convolution stage
    output = zeros(feature_map.shape)

    for map_num in range(feature_map.shape[-1]):
        for r in arange(0, feature_map.shape[0]):
            for c in arange(0, feature_map.shape[1]):
                output[r, c, map_num] = activation([feature_map[r, c, map_num], 0])

    return output


def pooling(feature_map, activation, size=2, stride=2):
    # Preparing the output of the pooling operation.
    pool_out = zeros((uint16((feature_map.shape[0] - size + 1) / stride + 1),
                      uint16((feature_map.shape[1] - size + 1) / stride + 1),
                      feature_map.shape[-1]))
    for map_num in range(feature_map.shape[-1]):
        r2 = 0
        for r in arange(0, feature_map.shape[0] - size + 1, stride):
            c2 = 0
            for c in arange(0, feature_map.shape[1] - size + 1, stride):
                pool_out[r2, c2, map_num] = activation([feature_map[r:r + size, c:c + size]])
                c2 = c2 + 1
            r2 = r2 + 1
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
                                                 [-1, 0, 1],
                                                 [-1, 0, 1]]])
            self.filter[1, :, :] = numpy.array([[[1, 1, 1],
                                                 [0, 0, 0],
                                                 [-1, -1, -1]]])
        else:
            self.filter = filter

    def activate(self):
        print('{} Activation'.format(self.layer_name))
        self.feature_maps = Convolution(self.__source, self.filter).feature_maps
        self.activated_maps = relu(self.feature_maps, self.__activation_func)
        self.reduced_maps = pooling(self.activated_maps, self.__activation_func)

    def plot_maps(self, rows, cols):
        fig, axes = plt.subplots(nrows=rows, ncols=cols)
        graphs(axes[0], self.feature_maps, '{}-Map{}', self.layer_name)
        graphs(axes[1], self.activated_maps, '{}-Map{}ReLU', self.layer_name)
        graphs(axes[2], self.reduced_maps, '{}-Map{}ReLUPool', self.layer_name)
        save_plot(fig, self.layer_name)
