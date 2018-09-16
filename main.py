import codecs
import json
from PIL import Image
from pylab import *
import glob, os

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy

jsonpickle_numpy.register_handlers()


class ParseImages:
    def __init__(self):
        self.__data = {
            'training': {
                'pos': {},
                'neg': {},
            },
            'test': {
                'normal': {},
                'scaled': {},
            }
        }

    def to_matrix(self, pic):
        return array(Image.open(pic))

    def to_matrix_arr(self, path, type, label):
        for infile in glob.glob(path):
            file, ext = os.path.splitext(infile)
            dictName = file.split('-')[1]
            self.__data[type][label][dictName] = self.to_matrix(infile)

        print('dataset->{}->{} parsed successfully'.format(type, label))

    def get_data(self):
        return self.__data

    def JSONify(self, path):
        print('saving to `{}`'.format(path))
        output = jsonpickle.encode(self.__data)
        json.dump(output, codecs.open(path, 'w', encoding='utf-8'))  ### this saves the array in .json format
        print('Done saving')


def handle_data(save_path):
    if os.path.isfile(save_path):
        print('`{}` already exist, I will just load in!'.format(save_path))
        input_file = codecs.open(save_path, 'r', encoding='utf-8').read()
        data = jsonpickle.decode(json.loads(input_file))
    else:
        print('`{}` does not exist, watch me create it!'.format(save_path))
        output = ParseImages()
        output.to_matrix_arr('./data_set/TrainImages/pos*.pgm', 'training', 'pos')
        output.to_matrix_arr('./data_set/TrainImages/neg*.pgm', 'training', 'neg')
        output.to_matrix_arr('./data_set/TestImages/test*.pgm', 'test', 'normal')
        output.to_matrix_arr('./data_set/TestImages_Scale/test*.pgm', 'test', 'scaled')
        output.JSONify(save_path)
        data = output.get_data()

    return data


def main():
    data = handle_data('./data.json')


main()
