import sys


def dimensions(tuple1, filter__, size=0, stripe= None):
    if stripe is not None:
        d1 = (tuple1[0] - filter__) / stripe + 1
        d2 = (tuple1[1] - filter__) / stripe + 1
        return tuple((d1, d2, size))

    d1 = tuple1[0] - filter__ + 1
    d2 = tuple1[1] - filter__ + 1

    return tuple((d1, d2, size))


def print_error(msg):
    print(msg)
    sys.exit()