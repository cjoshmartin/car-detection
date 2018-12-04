import numpy

from neuron.convolution import dot_product
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
        for i in range(self.filter.shape[-1]):
            curr_channel = self.filter[:, :, :, i]
            for j in range(curr_channel.shape[-1]):
                feature_maps[:, :, k] = dot_product(last_layer[:, :, j], curr_channel[:, :, j]) \
                                        # + feature_maps[:, :, k]
                k += 1

        return feature_maps

