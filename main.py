from neuron.neuron import Neuron, conv, relu
from utils.parse_images import handle_data, save_image
import numpy


# def create_filter

def main():
    data = handle_data('./data.json')
    training_data = data['training']
    test_data = data['test']

    activation = relu

    sample = training_data['pos']['0']
    prev_nuron = Neuron(sample, activation, 'Ly1')
    prev_nuron.activate()
    prev_nuron.plot_maps(3, prev_nuron.feature_maps.shape[2])

    meh = [[3,5,5], [1,7,7], [1,1,1]]

    for i in range(1, 2):
        nuron = Neuron(
            prev_nuron.reduced_maps,
            activation,
            'Ly{}'.format(i + 1),
            numpy.random.rand(
                meh[i-1][0],
                meh[i-1][1],
                meh[i-1][2],
                prev_nuron.reduced_maps.shape[-1]
            )
        )

        nuron.activate()
        nuron.plot_maps(3, nuron.feature_maps.shape[2])

        prev_nuron = nuron
    nuron

main()
