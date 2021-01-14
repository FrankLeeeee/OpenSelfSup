import os
import os.path as osp
import argparse
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str,
                        default='/4TB/sgli/projects/01-fyp-self-sup/datasets/imagenet/meta')
    parser.add_argument('--data-file', type=str)
    args = parser.parse_args()
    return args


def plot_dist(datafile):
    record = dict()

    with open(datafile, 'r') as f:
        for line in f:
            img_id, cls = line.strip().split()
            cls_list = record.get(cls, [])
            cls_list.append(img_id)
            record[cls] = cls_list

    count = [len(x) for x in record.values()]
    count.sort()

    plt.bar(range(len(count)), count)

    plt.xlabel('class')
    plt.ylabel('count')
    filename = osp.splitext(osp.basename(datafile))[0]
    plt.savefig('./{}_distribution.jpg'.format(filename))


def main():
    args = parse_args()

    datafile = osp.join(args.data_root, args.data_file)
    plot_dist(datafile)


if __name__ == "__main__":
    main()
