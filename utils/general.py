import numpy


def dimensions(tuple1, filter__, size=0, stripe=None):
    d1_sub = max(0, tuple1[0] - filter__)
    d2_sub = max(0, tuple1[1] - filter__)

    if stripe is not None:
        d1 = d1_sub / stripe + 1
        d2 = d2_sub / stripe + 1
        return tuple((d1, d2, size))

    d1 = d1_sub + 1
    d2 = d2_sub + 1

    return tuple((d1, d2, size))


def print_error(msg):
    raise Exception(msg)


def get_random_filter(shape, number_of_filters):
    return numpy.random.rand(
        shape[0],
        shape[1],
        shape[2],
        number_of_filters
    )