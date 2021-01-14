import os
import os.path as osp
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str,
                        default='/4TB/sgli/projects/01-fyp-self-sup/datasets/imagenet/meta')
    parser.add_argument("--val-file", type=str)
    parser.add_argument("--cls-mapping", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    return args


def get_subval(data_root, val_file, cls_mapping, output):
    datasource = osp.join(data_root, val_file)
    record = dict()

    with open(datasource, 'r') as f:
        for line in f:
            img_id, label = line.strip().split()
            img_list = record.get(label, [])
            img_list.append(img_id)
            record[label] = img_list

    output_file = osp.join(data_root, output)
    mapping_file = osp.join(data_root, cls_mapping)
    with open(output_file, 'w') as out_f:
        with open(mapping_file, 'r') as map_f:
            for line in map_f:
                new_cls, original_cls = line.strip().split()
                img_id_list = record[original_cls]

                for img_id in img_id_list:
                    out_f.write('{} {}\n'.format(img_id, new_cls))


def main():
    args = parse_args()
    get_subval(args.data_root, args.val_file, args.cls_mapping, args.output)


if __name__ == "__main__":
    main()
