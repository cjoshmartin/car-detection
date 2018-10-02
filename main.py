from utils.parse_images import handle_data, save_image
from pylab import *


def demations(tuple1, filter, size=0):
    d1 = tuple1[0] - filter[1] + 1
    d2 = tuple1[1] - filter[1] + 1

    return tuple((d1, d2, size))


def print_error(msg):
    print(msg)
    sys.exit()


def conv_(img, conv_filter):
    filter_size = conv_filter.shape[1]
    result = zeros((img.shape))
    # Looping through the image to apply the convolution operation.
    for r in uint16(arange(filter_size / 2.0,
                           img.shape[0] - filter_size / 2.0 + 1)):
        for c in uint16(arange(filter_size / 2.0,
                               img.shape[1] - filter_size / 2.0 + 1)):
            """
            Getting the current region to get multiplied with the filter.
            How to loop through the image and get the region based on 
            the image and filer sizes is the most tricky part of convolution.
            """
            curr_region = img[r - uint16(floor(filter_size / 2.0)):r + uint16(
                ceil(filter_size / 2.0)),
                          c - uint16(floor(filter_size / 2.0)):c + uint16(
                              ceil(filter_size / 2.0))]
            # Element-wise multipliplication between the current region and the filter.
            curr_result = curr_region * conv_filter
            conv_sum = sum(curr_result)  # Summing the result of multiplication.
            result[r, c] = conv_sum  # Saving the summation in the convolution layer feature map.

    # Clipping the outliers of the result matrix.
    final_result = result[uint16(filter_size / 2.0):result.shape[0] - uint16(filter_size / 2.0),
                   uint16(filter_size / 2.0):result.shape[1] - uint16(filter_size / 2.0)]
    return final_result


def conv(img, conv_filter):
    if len(img.shape) > 2 or len(conv_filter.shape) > 3:
        if img.shape[-1] != conv_filter.shape[-1]:
            print_error('Error: Number of channels in both image and filter must match.')
        if conv_filter.shape[1] != conv_filter.shape[2]:
            print_error('Error: Filter must be a square matrix. I.e. number of rows and columns must match.')
        if conv_filter.shape[2] % 2 == 0:
            print_error('Error: Filter must have an odd size. I.e. number of rows and columns must be odd.')

    number_of_filters = conv_filter.shape[0]
    feature_maps = zeros(demations(img.shape, conv_filter.shape, number_of_filters))

    for filter_num in range(number_of_filters):
        print("Filter ", filter_num + 1)
        curr_filter = conv_filter[filter_num, :]  # getting a filter from the bank.

        if len(curr_filter.shape) > 2:
            conv_map = conv_(img[:, :, 0], curr_filter[:, :, 0])
            for channel_number in range(1, curr_filter.shape[
                -1]):  # Convolving each channel with the image and summing the results.
                conv_map = conv_map + conv_(img[:, :, channel_number], curr_filter[:, :, channel_number])
        else:
            conv_map = conv_(img, curr_filter)  # single channel in filter
            feature_maps[:, :, filter_num] = conv_map

    return feature_maps


def main():
    data = handle_data('./data.json')
    training_data = data['training']
    test_data = data['test']


    test_filters = zeros((2, 3, 3))

    test_filters[0, :, :] = np.array([
        [1, 0, -1],
        [1, 0, -1],  # vertical edges
        [1, 0, -1]
    ])

    test_filters[1, :, :] = np.array([
        [1, 1, 1],
        [0, 0, 0],  # Horizontal edges
        [-1, -1, -1]
    ])

    sample = training_data['pos']['0']

    feature_map = conv(sample, test_filters)

    save_image(feature_map)


main()
