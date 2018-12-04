import numpy

from utils.general import print_error, dimensions


def dot_product(image, filter):
    # Element-wise multiplication between the current region and the filter.
    element_wise_multip = image * filter
    return numpy.sum(element_wise_multip) # summing the results of the multiplication


def convolution_range(half_filter_size,location, stride=1):
    __range = numpy.arange(half_filter_size, location - half_filter_size + 1, stride)
    return numpy.uint16(__range)


def conv_(img, conv_filter):
    filter_size = conv_filter.shape[1]
    half_filter_size = filter_size / 2.0
    result = numpy.zeros(img.shape)


    row_range = convolution_range(half_filter_size, img.shape[0])
    column_range = convolution_range(half_filter_size, img.shape[1])

    for row in row_range:
        for column in column_range:
            floor_of_filter = numpy.uint16(numpy.floor(half_filter_size))
            ceil_of_filter = numpy.uint16(numpy.ceil(half_filter_size))

            curr_region = img[
                          row - floor_of_filter : row + ceil_of_filter,
                          column - floor_of_filter : column + ceil_of_filter
                          ]

            result[row, column] = dot_product(curr_region, conv_filter)  # Saving the summation in the convolution layer feature map.

    # Clipping the outliers of the result matrix.
    trimmed_zeros = numpy.uint16(half_filter_size) # converting half_filter_size to an int
    return result[
                   trimmed_zeros: result.shape[0] - trimmed_zeros,
                   trimmed_zeros: result.shape[1] - trimmed_zeros
                   ]


def conv(img, conv_filter):
    # An empty feature map to hold the output of convoluting the filter(s) with the image.
    d_out = dimensions(img.shape, conv_filter.shape[1], conv_filter.shape[0])
    feature_maps = numpy.zeros(d_out)

    # Convolving the image by the filter(s).
    for filter_num in range(conv_filter.shape[0]):
        print("Filter ", filter_num + 1)
        current_filter = conv_filter[filter_num, :]  # getting a filter from the bank.

        if len(current_filter.shape) > 2:
            convolution_map = conv_(img[:, :, 0], current_filter[:, :, 0])  # Array holding the sum of all feature maps.
            for channel_number in range(1, current_filter.shape[-1]):  # Convolving each channel with the image and summing the results.
                convolution_map = convolution_map + conv_(img[:, :, channel_number], current_filter[:, :, channel_number])
        else:
            convolution_map = conv_(img, current_filter)

        feature_maps[:, :, filter_num] = convolution_map  # Holding feature map with the current filter.
    return feature_maps  # Returning all feature maps.