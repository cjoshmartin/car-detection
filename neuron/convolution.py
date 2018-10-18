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

    # Looping through the image to apply the convolution operation.
    for row in row_range:
        for column in column_range:
            """
            Getting the current region to get multiplied with the filter.
            How to loop through the image and get the region based on 
            the image and filer sizes is the most tricky part of convolution.
            """
            floor_of_filter = numpy.uint16(numpy.floor(half_filter_size))
            ceil_of_filter = numpy.uint16(numpy.ceil(half_filter_size))

            curr_region = img[
                          row - floor_of_filter : row + ceil_of_filter,
                          column - floor_of_filter : column + ceil_of_filter
                          ]

            result[row, column] = dot_product(curr_region, conv_filter)  # Saving the summation in the convolution layer feature map.

    # Clipping the outliers of the result matrix.
    hfs_int = numpy.uint16(half_filter_size) # converting half_filter_size to an int
    return result[
                   hfs_int: result.shape[0] - hfs_int,
                   hfs_int: result.shape[1] - hfs_int
                   ]


def conv(img, conv_filter):
    if len(img.shape) > 2 or len(conv_filter.shape) > 3:  # Check if number of image channels matches the filter depth.
        if img.shape[-1] != conv_filter.shape[-1]:
            print_error("Error: Number of channels in both image and filter must match.")
    if conv_filter.shape[1] != conv_filter.shape[2]:  # Check if filter dimensions are equal.
        print_error('Error: Filter must be a square matrix. I.e. number of rows and columns must match.')
    if conv_filter.shape[1] % 2 == 0:  # Check if filter diemnsions are odd.
        print_error('Error: Filter must have an odd size. I.e. number of rows and columns must be odd.')

    # An empty feature map to hold the output of convolving the filter(s) with the image.
    d_out = dimensions(img.shape, conv_filter.shape[1], conv_filter.shape[0]) # TODO look at, maybe setting the wrong demantions
    feature_maps = numpy.zeros(d_out)

    # Convolving the image by the filter(s).
    for filter_num in range(conv_filter.shape[0]):
        print("Filter ", filter_num + 1)
        curr_filter = conv_filter[filter_num, :]  # getting a filter from the bank.
        """ 
        Checking if there are mutliple channels for the single filter.
        If so, then each channel will convolve the image.
        The result of all convolutions are summed to return a single feature map.
        """
        if len(curr_filter.shape) > 2:
            conv_map = conv_(img[:, :, 0], curr_filter[:, :, 0])  # Array holding the sum of all feature maps.
            for ch_num in range(1, curr_filter.shape[-1]):  # Convolving each channel with the image and summing the results.
                conv_map = conv_map + conv_(img[:, :, ch_num], curr_filter[:, :, ch_num])
        else:  # There is just a single channel in the filter.
            conv_map = conv_(img, curr_filter)
        feature_maps[:, :, filter_num] = conv_map  # Holding feature map with the current filter.
    return feature_maps  # Returning all feature maps.