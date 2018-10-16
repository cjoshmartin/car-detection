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
    sample_2 = [ training_data['pos'], training_data['neg']]
    prev_neuron = Neuron(sample, activation, 'Ly1')
    prev_neuron.set_should_pooling(False)
    prev_neuron.activate()
    prev_neuron.plot_maps(0)

    network = [prev_neuron]

    meh = [[3,5,5], [1,7,7], [1,1,1]]

    for i in range(1, 3):
        neuron = Neuron(
            prev_neuron.get_map(),
            activation,
            'Ly{}'.format(i + 1),
            numpy.random.rand(
                meh[i-1][0],
                meh[i-1][1],
                meh[i-1][2],
                prev_neuron.get_map().shape[-1]
            )
        )
        neuron.set_should_pooling(False)
        network.append(neuron)

        neuron.activate()
        neuron.plot_maps(0)

        prev_neuron = neuron

    epoch = 0
    while True:
        print('\nEpoch #{}\n'.format(epoch))

        for i in range(len(sample_2[1])):

            has_a_car = (i + epoch ) % 2
            sample = sample_2[has_a_car]['{}'.format(i)]

            for j in range(len(network)):
                if j == 0:
                    network[j].set_image(sample)
                else:
                    network[j].set_image(network[j-1].get_map())

                network[j].activate()

                network[j].plot_maps(i)

            if i == 10:
                break
        epoch = epoch + 1
main()
