import os
import os.path as osp
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='/4TB/sgli/projects/01-fyp-self-sup/datasets/imagenet/meta')
    parser.add_argument('--data-file', type=str)
    parser.add_argument('--label', action='store_true', default=False)
    args = parser.parse_args()
    return args


def get_stats(datafile, with_label):
    record = dict()

    with open(datafile, 'r') as f:
        for line in f:
            if with_label:
                img_id, cls = line.strip().split()
            else:
                img_id = line.strip()
                cls_list = '-1'
            cls_list = record.get(cls, [])
            cls_list.append(img_id)
            record[cls] = cls_list

    record_items = list(record.items())
    record_items.sort(key=lambda x: len(x[1]))

    print('Number of clases: {}'.format(len(record_items)))
    print('Number of items in top 10 classes: {}'.format(
        [len(item[1]) for item in record_items[-10:]][::-1]
        ))
    print('Number of items in bottomw 10 classes: {}'.format(
        [len(item[1]) for item in record_items[:10]][::-1]
        ))

            
def main():
    args = parse_args()

    datafile = osp.join(args.data_root, args.data_file)
    get_stats(datafile, args.label)

if __name__ == "__main__":
    main()





