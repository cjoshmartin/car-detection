import codecs
import json
from PIL import Image
from pylab import *
import glob, os
def toMatrix(pic):
    return array(Image.open(pic))
def main():
    dataSet = {
        'training': {
            'pos': {},
            'neg': {},
        },
        'test': {
            'normal': {},
            'scaled': {},
        }
    }

    for infile in glob.glob("./data_set/TrainImages/pos*.pgm"):
        file, ext = os.path.splitext(infile)
        dictName = file.split('-')[1]
        dataSet['training']['pos'][dictName] = toMatrix(infile).tolist()

    for infile in glob.glob("./data_set/TrainImages/neg*.pgm"):
        file, ext = os.path.splitext(infile)
        dictName = file.split('-')[1]
        dataSet['training']['neg'][dictName] = toMatrix(infile).tolist()

    for infile in glob.glob("./data_set/TestImages/test*.pgm"):
        file, ext = os.path.splitext(infile)
        dictName = file.split('-')[1]
        dataSet['test']['normal'][dictName] = toMatrix(infile).tolist()

    for infile in glob.glob("./data_set/TestImages_Scale/test*.pgm"):
        file, ext = os.path.splitext(infile)
        dictName = file.split('-')[1]
        dataSet['test']['scaled'][dictName] = toMatrix(infile).tolist()

    json.dump(dataSet, codecs.open("./data.JSON", 'w', encoding='utf-8'), separators=(',', ':'), indent=4)  ### this saves the array in .json format

main()