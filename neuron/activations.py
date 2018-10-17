import numpy
from numpy.core.multiarray import zeros, arange


def relu(feature_map):  # Activation function, normalize of what is passed from the convolution stage
    output = zeros(feature_map.shape)

    for map_num in range(feature_map.shape[-1]):
        for row in arange(0, feature_map.shape[0]):
            for column in arange(0, feature_map.shape[1]):
                output[row, column, map_num] = numpy.max([0, feature_map[row, column, map_num]])

    return output