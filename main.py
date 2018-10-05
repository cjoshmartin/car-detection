from neuron.neuron import Neuron, Convolution
from utils.parse_images import handle_data, save_image
import numpy


# def create_filter

def main():
    data = handle_data('./data.json')
    training_data = data['training']
    test_data = data['test']

    sample = training_data['pos']['0']
    nuron = Neuron(sample, numpy.max, 'Ly1')
    nuron.activate()
    nuron.plot_maps(3, 2)


    # next_nuron = Neuron(
    #     nuron.reduced_maps,
    #     numpy.max,'Ly2',
    #     numpy.random.rand(3, 5, 5, nuron.reduced_maps.shape[-1])
    # )
    # next_nuron.activate()
    # next_nuron.plot_maps(3,3)
    l2_filter = numpy.random.rand(3, 5, 5, nuron.reduced_maps.shape[-1])
    print("\n**Working with conv layer 2**")
    l2_feature_map =Convolution(nuron.reduced_maps, l2_filter).feature_maps

    save_image('testy',l2_feature_map[:,:,0])

main()
