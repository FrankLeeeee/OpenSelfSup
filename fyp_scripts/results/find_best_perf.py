import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, help='path to train log')
    return parser.parse_args()


def read_stasts(file_path):
    max_top1_acc = 0
    max_top5_acc = 0

    with open(file_path, 'r') as f:
        lines = f.readlines()

        for i in range(len(lines)):
            if 'head0_top1' in lines[i] and 'Epoch' not in lines[i]:
                top1_acc = float(lines[i].split(':')[-1])

                if top1_acc > max_top1_acc:
                    max_top1_acc = top1_acc

                    top5_acc = float(lines[i+1].split(':')[-1])
                    max_top5_acc = top5_acc

    print('top 1 acc: {}, top5 acc: {}'.format(max_top1_acc, max_top5_acc))


if __name__ == '__main__':
    args = parse_args()
    read_stasts(args.f)
