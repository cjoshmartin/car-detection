from neuron.neuron import Neuron, relu
from utils.parse_images import handle_data
import numpy


# def create_filter

def main():
    data = handle_data('./data.json')
    training_data = data['training']
    test_data = data['test']

    activation = relu

    sample = [ training_data['pos'], training_data['neg']]
    prev_neuron = Neuron(training_data['pos']['0'], activation, 'Ly1')
    prev_neuron.activate()
    network = [prev_neuron]

    meh = [[3,5,5], [1,7,7], [1,1,1]]

    for i in range(len(meh)):
        neuron = Neuron(
            prev_neuron.get_map(),
            activation,
            'Ly{}'.format(i + 1),
            numpy.random.rand(
                meh[i][0],
                meh[i][1],
                meh[i][2],
                prev_neuron.get_map().shape[-1] # TODO: find a way to decomple this, cause to have to run activation function to have shape
            )
        )
        # neuron.set_should_pooling(False)
        neuron.activate()
        network.append(neuron)

        prev_neuron = neuron

    epoch = 0
    while True:
        print('\nEpoch #{}\n'.format(epoch))

        for i in range(len(sample[1])):

            has_a_car = (i + epoch ) % 2
            sample = sample[has_a_car]['{}'.format(i)]

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
