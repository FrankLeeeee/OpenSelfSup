import os
import os.path as osp
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", type=str, required=True)
    parser.add_argument("--train_root", type=str, required=True)
    args = parser.parse_args()
    return args


def verify_exist(train_list, train_root):
    total = 0
    missing = 0

    with open(train_list, 'r') as f:
        for line in f:
            img_path = osp.join(train_root, line.strip())
            total += 1

            if not osp.exists(img_path):
                print("Not found: {}".format(img_path))
                missing += 1

    print("Total images: {}".format(total))
    print("Missing images: {}".format(missing))


def main():
    args = parse_args()
    verify_exist(args.train_list, args.train_root)


if __name__ == "__main__":
    main()
