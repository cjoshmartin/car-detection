from network.network import Network
import utils.activations as activations
from utils.parse_images import handle_data


def main():
    data = handle_data('./data.json')

    training_data = data['training']
    test_data = data['test']
    sample = [training_data['pos'], training_data['neg']]
    activation = activations.relu

    layer_config = [
        [3, 5, 5]
    ]

    cnn_network = Network(activation, training_data['pos']['550'], layer_config)

    epoch = 0

    # while True:
    print('\nEpoch #{}\n'.format(epoch))
    cnn_network.train(sample, epoch, 10)
    epoch += 1


main()
