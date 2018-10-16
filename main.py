from network.network import Network
from neuron.neuron import Neuron, relu
from utils.parse_images import handle_data
import numpy


# def create_filter

def main():
    data = handle_data('./data.json')
    training_data = data['training']
    test_data = data['test']
    sample = [training_data['pos'], training_data['neg']]
    activation = relu

    layer_config = [[3,5,5], [1,7,7], [1,1,1]]

    cnn_network = Network(activation, training_data['pos']['0'], layer_config)

    epoch = 0

    while True:
        print('\nEpoch #{}\n'.format(epoch))
        cnn_network.train(sample, epoch, 10)
        epoch += 1

main()
