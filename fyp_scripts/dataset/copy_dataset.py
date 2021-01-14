import argparse
import os
import os.path as osp
import shutil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        help='input text file', required=True)
    parser.add_argument('-r', '--root', type=str,
                        help='input text file', required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    return parser.parse_args()


def copy_dataset(data_root, data_file, destination):
    with open(data_file, 'r') as f:
        for line in f:
            file_path = line.strip()
            full_file_path = osp.join(data_root, file_path)

            output_path = osp.join(destination, file_path)
            dir_output_path = osp.dirname(output_path)

            if not osp.exists(dir_output_path):
                os.mkdir(dir_output_path)

            shutil.copy(full_file_path, dir_output_path)


def main():
    args = parse_args()
    copy_dataset(args.root, args.input, args.output)


if __name__ == "__main__":
    main()
