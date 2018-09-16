import codecs
import json
from PIL import Image
from pylab import *
import glob, os

class parseImages():
    def __init__(self):
        self.__dataSet = {
        'training': {
            'pos': {},
            'neg': {},
        },
        'test': {
            'normal': {},
            'scaled': {},
        }
    }

    def toMatrix(self,pic):
        return array(Image.open(pic))

    def toMatrixArr(self,path, type, label):
        for infile in glob.glob(path):
            file, ext = os.path.splitext(infile)
            dictName = file.split('-')[1]
            self.__dataSet[type][label][dictName] = self.toMatrix(infile).tolist()

        print('dataset->{}->{} parsed successfully'.format(type,label))

    def getData(self):
            return self.__dataSet

    def JSONify(self,path):
        print('saving to `{}`'.format(path))
        json.dump(self.__dataSet, codecs.open(path, 'w', encoding='utf-8'), separators=(',', ':'), indent=4)  ### this saves the array in .json format
        print('Done saving')

class handleData():
    def __init__(self):
        output = parseImages()
        output.toMatrixArr('./data_set/TrainImages/pos*.pgm','training','pos')
        output.toMatrixArr('./data_set/TrainImages/neg*.pgm','training','neg')
        output.toMatrixArr('./data_set/TestImages/test*.pgm','test','normal')
        output.toMatrixArr('./data_set/TestImages_Scale/test*.pgm','test','scaled')
        output.JSONify('./data.json')

def main():
    data = handleData()


main()