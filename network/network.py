from neuron.neuron import Neuron
import numpy

from utils.activations import softmax


def get_random_filter(shape, number_of_filters):
    return numpy.random.rand(
        shape[0],
        shape[1],
        shape[2],
        number_of_filters
    )


class Network:

    def __init__(self, activation, sample_image, layer_configuration, should_pool=True):

        self.__prev_neuron = Neuron(sample_image, activation, 'Ly1')
        self.__prev_neuron.set_should_pooling(should_pool)
        self.__prev_neuron.activate()
        self.__network = [self.__prev_neuron]

        size_of_config = len(layer_configuration)
        for i in range(size_of_config):
            neuron = Neuron(
                self.__prev_neuron.get_map(),
                activation,
                'Ly{}'.format(i + 2),
                get_random_filter(
                    layer_configuration[i],
                    self.__prev_neuron.get_map().shape[-1]
                    # TODO: find a way to decompile this, cause to have to run activation function to have shape
                )
            )
            neuron.set_should_pooling(should_pool)
            neuron.activate()
            self.__network.append(neuron)

            self.__prev_neuron = neuron

        tacos = self.__prev_neuron.get_map()
        neuron = Neuron( # Fully Connected layer
           tacos,
            softmax,  # classifier
            'Full Connected Layer',
            get_random_filter(
                tacos.shape, # TODO: Transpose this shit, motherfucking bitch
                self.__prev_neuron.get_map().shape[-1]
            )
        )

        neuron.set_should_pooling(False) # Do not want down-sampling on output layer
        neuron.activate()   # Do I need this
        self.__network.append(neuron)

    def train(self, input_data, cur_epoch, batch_size):
        for i in range(len(input_data[1])):

            has_a_car = (i + cur_epoch) % 2
            data = input_data[has_a_car]['{}'.format(i)]

            for j in range(len(self.__network)):
                if j == 0:
                    self.__network[j].set_image(data)
                else:
                    self.__network[j].set_image(self.__network[j - 1].get_map())

                self.__network[j].activate()

                # self.__network[j].plot_maps(i)

            if i == batch_size:
                break
        classifation = self.__network[len(self.__network) - 1].get_map()
        print(classifation)

    def test(self):  # TODO
        pass
