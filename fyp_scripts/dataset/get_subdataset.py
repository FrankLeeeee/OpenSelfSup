import os
import os.path as osp
import argparse
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str,
                        default='/4TB/sgli/projects/01-fyp-self-sup/datasets/imagenet/meta')
    parser.add_argument('--data-file', type=str)
    parser.add_argument('--percent', type=int)
    parser.add_argument('--cls-interval', type=int)
    args = parser.parse_args()
    return args


def get_subdataset(data_root, data_file, percent, cls_interval):
    datasource = osp.join(data_root, data_file)
    record = dict()

    with open(datasource, 'r') as f:
        for line in f:
            img_id, cls = line.strip().split()
            img_list = record.get(cls, [])
            img_list.append(img_id)
            record[cls] = img_list

    record_items = list(record.items())
    record_items.sort(key=lambda x: len(x[1]))

    # keep some classes only
    record_items = record_items[::cls_interval]
    record_items = [
        (item[0], random.sample(item[1], int(len(item[1]) * percent / 100))) for item in record_items
    ]

    output_path = osp.join(data_root, 'subdataset')
    if not osp.exists(output_path):
        os.mkdir(output_path)
    output_file = osp.join(output_path, '{}_{}percent_{}interval.txt'.format(
        data_file.split('.')[0], percent, cls_interval
    ))

    label_mapping = osp.join(output_path, '{}_{}percent_{}interval_label_mapping.txt'.format(
        data_file.split('.')[0], percent, cls_interval
    ))

    with open(output_file, 'w') as f:
        with open(label_mapping, 'w') as label_f:
            cls_id = 0

            for item in record_items:
                original_cls = item[0]
                label_f.write('{} {}\n'.format(cls_id, original_cls))
                for img_id in item[1]:
                    f.write('{} {}\n'.format(img_id, cls_id))
                cls_id += 1


def main():
    args = parse_args()
    get_subdataset(args.data_root, args.data_file,
                   args.percent, args.cls_interval)


if __name__ == "__main__":
    main()
