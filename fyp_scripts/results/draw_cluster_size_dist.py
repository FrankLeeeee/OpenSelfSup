import argparse
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f',
        '--file',
        type=str,
        help='path to the cluster file',
        required=True)
    parser.add_argument(
        '-n', '--num', type=int, help='number of classes', required=True)
    parser.add_argument(
        '-t', '--title', type=str, help='title of the chart', required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    cluster = np.load(args.file)
    c = Counter(cluster)
    items = list(c.values())
    items.sort()
    items = [0] * (args.num - len(items)) + items
    plt.plot(range(args.num), items, 'bo')
    plt.xlabel('cluster ')
    plt.ylabel('size')
    plt.title(args.title)
    plt.savefig('./plots/{}.jpg'.format(args.title))


if __name__ == '__main__':
    main()
