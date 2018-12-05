import numpy

from neuron.convolution import dot_product, conv_
from neuron.neuron import Neuron
import utils.activations as activations
import utils.general as general


def fc_layer(source):
    init_filter = general.get_random_filter(
        source.shape,
        source.shape[-1]
    )

    return FullConnected(
        source,
        activations.softmax,
        'Full Connected Layer',
        init_filter
    )


class FullConnected(Neuron):
    def __int__(self):
        pass

    def conv(self, bias):  # function overloading

        last_layer = self._Neuron__source
        d_out = general.dimensions(last_layer.shape, self.filter.shape[1],
                                   self.filter.shape[0])
        feature_maps = numpy.zeros(d_out)
        k = 0
        number_of_channels = self.filter.shape[-1]
        for i in range(number_of_channels):
            curr_channel = self.filter[:, :, :, i]
            for j in range(curr_channel.shape[-1]):
                feature_maps[:, :, k] = dot_product(last_layer[:, :, j], curr_channel[:, :, j]) \
                                        # + feature_maps[:, :, k]
                if k < number_of_channels:
                    k += 1
                else:
                    k = 0

        return feature_maps

    # def conv(self, bias):
    #     img = self._Neuron__source
    #     conv_filter = self.filter
    #     # def conv(self, bias):  # function overloading
    #     # An empty feature map to hold the output of convolving the filter(s) with the image.
    #     the_shape = (1,
    #                  1,
    #                  conv_filter.shape[2])
    #     feature_maps = numpy.zeros(the_shape)
    #
    #     # Convolving the image by the filter(s).
    #     for filter_num in range(conv_filter.shape[0]):
    #         print("Filter ", filter_num + 1)
    #         curr_filter = conv_filter[filter_num, :]  # getting a filter from the bank.
    #         """
    #         Checking if there are mutliple channels for the single filter.
    #         If so, then each channel will convolve the image.
    #         The result of all convolutions are summed to return a single feature map.
    #         """
    #         if len(curr_filter.shape) > 2:
    #             conv_map = conv_(img[:, :, 0], curr_filter[:, :, 0])  # Array holding the sum of all feature maps.
    #             for ch_num in range(1, curr_filter.shape[
    #                 -1]):  # Convolving each channel with the image and summing the results.
    #                 conv_map = conv_map + conv_(img[:, :, ch_num],
    #                                             curr_filter[:, :, ch_num])
    #         else:  # There is just a single channel in the filter.
    #             conv_map = conv_(img, curr_filter)
    #         feature_maps[:, :, filter_num] = conv_map  # Holding feature map with the current filter.
    #     return feature_maps  # Returning all feature maps.

