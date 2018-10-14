import sys


def demations(tuple1, filter, size=0):
    d1 = tuple1[0] - filter[1] + 1
    d2 = tuple1[1] - filter[1] + 1

    return tuple((d1, d2, size))


def print_error(msg):
    print(msg)
    sys.exit()