import numpy
import glob
from PIL import Image
from resizeimage import resizeimage


def resize(input_image, save_to):
    img = Image.open(input_image)
    img = resizeimage.resize_cover(img, [100, 40])
    img.save(save_to, img.format)


def copy_images(path, training, testing, stride, name, number, test_num):
    image_arr = glob.glob(path)
    image_arr_length = len(image_arr)
    i = 1
    for selected_file in numpy.arange(0, image_arr_length - 1, stride):
        training_output_number = number + i
        testing_output_number = test_num + i

        training_loc = '{}/{}-{}.jpg'.format(training, name, training_output_number)
        resize(image_arr[selected_file], training_loc)

        test_loc = '{}/testing-{}.jpg'.format(testing, testing_output_number)
        resize(image_arr[selected_file + 1], test_loc)
        i += 1
        print('{} --> {} '.format(image_arr[selected_file], training_loc))

    print()

    return (number + i), (test_num + i)


new_training, new_testing = copy_images(
    '/Users/josh/Desktop/school/Senior/ECE_570/datasets/cars_brad/*.jpg',
    '../data_set/caltech/training',
    '../data_set/caltech/testing',
    4,
    'pos',
    549,
    169
)
copy_images(
    '/Users/josh/Desktop/school/Senior/ECE_570/datasets/cars_markus/*.jpg',
    '../data_set/caltech/training',
    '../data_set/caltech/testing',
    4,
    'pos',
    new_training,
    new_testing
)
