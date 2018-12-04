import math
import numpy
from numpy.core.multiarray import zeros, arange


def relu(feature_map):  # Activation function, normalize of what is passed from the convolution stage
    shape = feature_map.shape
    output = zeros(shape)

    for channel_number in range(shape[-1]):
        for row in arange(0, shape[0]):
            for column in arange(0, shape[1]):
                output[row, column, channel_number] = numpy.max([0, feature_map[row, column, channel_number]])

    return output


def sigmoid(feature_map):
    shape = feature_map.shape
    output = zeros(shape)

    for channel_number in range(shape[-1]):
        for row in arange(0, shape[0]):
            for column in arange(0, shape[1]):
                output[row, column, channel_number] = 1 / (1 + math.exp(feature_map[row, column, channel_number]))


def softmax(feature_map):
    if False: # TODO: Hot fix, remove once data is normalized
        exponential = numpy.exp(feature_map)
        return exponential / numpy.sum(exponential)

    return numpy.sum(feature_map)
