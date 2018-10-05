from neuron.neuron import relu, pooling, Convolution, Neuron
from utils.parse_images import handle_data
from pylab import *
import numpy
import matplotlib.pyplot as plt


# def create_filter

def main():
    data = handle_data('./data.json')
    training_data = data['training']
    test_data = data['test']

    sample = training_data['pos']['0']
    nuron = Neuron(sample, numpy.max, 'Ly1')
    nuron.plot_maps(3, 2)


main()
