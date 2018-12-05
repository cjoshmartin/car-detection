import codecs
import json
import numpy
from PIL import Image
from pylab import *
import glob, os

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy

from utils.ploting import  plot, save_plot
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
        return array(Image.open(pic).convert('L'))

    def normalize(self, matrix):
        matrix -= numpy.mean(matrix, axis=0, dtype=numpy.uint8)
        cov = numpy.dot(matrix.T, matrix) / matrix.shape[0]
        U, S, V = np.linalg.svd(cov)
        Xrot = np.dot(matrix, U)
        Xrot_reduced = np.dot(matrix, U[:, :100])
        # whiten the data:
        # divide by the eigenvalues (which are square roots of the singular values)
        Xwhite = Xrot / np.sqrt(S + 1e-5)
        return Xwhite

    def to_matrix_arr(self, path, type, label):
        for infile in glob.glob(path):
            file, ext = os.path.splitext(infile)
            dictName = file.split('-')[1]
            # tacos = self.to_matrix(infile)
            normalized_data = self.normalize( self.to_matrix(infile) )

            self.__data[type][label][dictName] = normalized_data

        print('dataset->{}->{} parsed successfully'.format(type, label))

    def get_data(self):
        return self.__data

    def JSONify(self, path):
        print('saving to `{}`'.format(path))
        output = jsonpickle.encode(self.__data)
        json.dump(output, codecs.open(path, 'w', encoding='utf-8'))  ### this saves the array in .json format
        print('Done saving')

    def append(self, loc, type, is_car, dictName):
        self.__data[type][is_car][dictName] = self.to_matrix(loc)

def handle_data(save_path):
    if os.path.isfile(save_path):
        print('`{}` already exist, I will just load in!'.format(save_path))
        input_file = codecs.open(save_path, 'r', encoding='utf-8').read()
        data = jsonpickle.decode(json.loads(input_file))
    else:
        print('`{}` does not exist, watch me create it!'.format(save_path))
        output = ParseImages()
        output.to_matrix_arr('./data_set/uiuc/TrainImages/pos*.pgm', 'training', 'pos')
        output.to_matrix_arr('./data_set/caltech/training/pos*.jpg', 'training', 'pos')
        output.to_matrix_arr('./data_set/uiuc/TrainImages/neg*.pgm', 'training', 'neg')
        output.to_matrix_arr('./data_set/uiuc/TestImages/test*.pgm', 'test', 'normal')
        output.to_matrix_arr('./data_set/caltech/testing/test*.jpg', 'test', 'normal')
        output.to_matrix_arr('./data_set/uiuc/TestImages_Scale/test*.pgm', 'test', 'scaled')
        output.JSONify(save_path)
        data = output.get_data()

    return data

def save_image(name, source):
    __fig, __axes = plt.subplots(nrows=1, ncols=1)
    plot(__axes, source, name)
    save_plot(__fig, name)




