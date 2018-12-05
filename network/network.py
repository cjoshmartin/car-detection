import math

import numpy

from neuron.full_connected import fc_layer
from neuron.neuron import Neuron
import utils.general as general


def create_indacation(is_a_car):
    if is_a_car > 0:
        return "Has a car "

    return "doesn\'t have a car"


class Network:

    def __init__(self, activation, sample_image, layer_configuration, should_pool=True):

        self.__prev_neuron = Neuron(sample_image, activation, 'Ly1')
        self.__prev_neuron.set_should_pooling(should_pool)
        self.__prev_neuron.activate()
        self.__prev_neuron.plot_maps(0)
        self.__network = [self.__prev_neuron]

        size_of_config = len(layer_configuration)
        for i in range(size_of_config):
            neuron = Neuron(
                self.__prev_neuron.get_map(),
                activation,
                'Ly{}'.format(i + 2),
                general.get_random_filter(
                    layer_configuration[i],
                    self.__prev_neuron.get_map().shape[-1]
                )
            )
            neuron.set_should_pooling(should_pool)
            neuron.activate()
            neuron.plot_maps(i)
            self.__network.append(neuron)

            self.__prev_neuron = neuron

        tacos = self.__prev_neuron.get_map()
        neuron = fc_layer(tacos)  # Full Connected Layer

        neuron.set_should_pooling(False)  # Do not want down-sampling on output layer
        neuron.activate()
        self.__network.append(neuron)

    def train(self, input_data, cur_epoch, batch_size):
        error_sum = 0

        for i in range(len(input_data[1])):

            has_a_car = (i + cur_epoch) % 2
            data = input_data[has_a_car]['{}'.format(i)]

            for j in range(len(self.__network) - 1):
                if j == 0:
                    # data -= numpy.mean(data, axis=0)
                    self.__network[j].set_image(data)
                else:
                    self.__network[j].set_image(self.__network[j - 1].get_map())

                self.__network[j].activate()

                self.__network[j].plot_maps(i)

            classifation = self.__network[len(self.__network) - 1].get_map()

            correct_answer = create_indacation(has_a_car)
            guess = create_indacation(classifation)

            print(
                "\n========================================\n"
                "the correct answer for this images is: {}"
                "\nhowever the network has guessed: {}"
                "\nleading to a error of {}"
                "\n========================================\n".format(correct_answer, guess, .5 * math.pow((has_a_car - classifation), 1)))

            if i == batch_size:
                avg_error =  error_sum / batch_size

                print ("\n==============================\n"
                       "YELLING AT THE TOP OF MY LUNGS"
                       "\n{}\n".format(avg_error))

            else:
                error_sum += .5 * math.pow((has_a_car - classifation), 1)

    def test(self):  # TODO
        pass
