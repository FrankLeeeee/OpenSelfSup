import matplotlib.pyplot as plt

SIMCLR_Y_TOP1 = [0.5712, 0.6240, 0.64580, 0.6548, 0.6558]
ODC_Y_TOP1 = [0.3846, 0.4578, 0.5116, 0.5506, 0.5750]
CONODC_V24_Y_TOP1 = [0.5674, 0.6296, 0.6530, 0.6660, 0.6660]

SIMCLR_Y_TOP5 = [0.8010, 0.8402, 0.8584, 0.8658, 0.8654]
ODC_Y_TOP5 = [0.6458, 0.7176, 0.7684, 0.7948, 0.8102]
CONODC_V24_Y_TOP5 = [0.8010, 0.8466, 0.8656, 0.8676, 0.8701]

X = list(range(40, 201, 40))


def main():
    plt.plot(X, SIMCLR_Y_TOP1, label="SimCLR")
    plt.plot(X, ODC_Y_TOP1, label="ODC")
    plt.plot(
        X, CONODC_V24_Y_TOP1, label="Contrastive ODC V3")  # For V24 in exp log

    plt.xlabel('epoch')
    plt.ylabel('accurary')
    plt.title('ImageNet Classification Top 1 Accuracy')
    plt.legend()

    plt.savefig('./plots/{}.jpg'.format(
        'ImageNet Classification Top 1 Accuracy By Epoch'))

    plt.clf()

    plt.plot(X, SIMCLR_Y_TOP5, label="SimCLR")
    plt.plot(X, ODC_Y_TOP5, label="ODC")
    plt.plot(
        X, CONODC_V24_Y_TOP5, label="Contrastive ODC V3")  # For V24 in exp log

    plt.xlabel('epoch')
    plt.ylabel('accurary')
    plt.title('ImageNet Classification Top 5 Accuracy')
    plt.legend()

    plt.savefig('./plots/{}.jpg'.format(
        'ImageNet Classification Top 5 Accuracy By Epoch'))


if __name__ == '__main__':
    main()
