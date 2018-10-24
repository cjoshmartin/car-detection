from network.network import Network
from utils.activations import relu
from utils.parse_images import handle_data


# def create_filter

def main():
    data = handle_data('./data.json')

    training_data = data['training']
    test_data = data['test']
    sample = [training_data['pos'], training_data['neg']]
    activation = relu

    layer_config = [[3, 5, 5]]

    cnn_network = Network(activation, training_data['pos']['550'], layer_config)
    #
    # epoch = 0
    #
    # # while True:
    # print('\nEpoch #{}\n'.format(epoch))
    # cnn_network.train(sample, epoch, 2)
    # epoch += 1


main()
