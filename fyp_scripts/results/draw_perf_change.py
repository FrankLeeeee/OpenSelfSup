import matplotlib.pyplot as plt

MINCLUSTER = [32, 64, 128, 258]
NUM_CLUSTER_TOP_1 = [0.6644, 0.6650, 0.65440, 0.6596]
NUM_CLUSTER_TOP_5 = [0.86560, 0.8680, 0.86660, 0.8654]

NUMCLUSTER = [100, 200, 400, 800]
MIN_CLUSTER_TOP_1 = [0.6644, 0.6596, 0.66480, 0.6606]
MIN_CLUSTER_TOP_5 = [0.86560, 0.86540, 0.87260, 0.86740]


def main():
    for idx, y in enumerate([NUM_CLUSTER_TOP_1, NUM_CLUSTER_TOP_5]):
        plt.plot(MINCLUSTER, y)
        plt.xlabel('min cluster size')
        plt.ylabel('accuracy')

        if idx == 0:
            acc = 'Top 1'
        else:
            acc = 'Top 5'

        title = 'ImageNet Classification {} Accurary With 100 clusters'.format(
            acc)
        plt.title(title)
        plt.savefig('./plots/{}.jpg'.format(title))
        plt.clf()

    for idx, y in enumerate([MIN_CLUSTER_TOP_1, MIN_CLUSTER_TOP_5]):
        plt.plot(NUMCLUSTER, y)
        plt.xlabel('number of clusters')
        plt.ylabel('accuracy')

        if idx == 0:
            acc = 'Top 1'
        else:
            acc = 'Top 5'

        title = 'ImageNet Classification {} Accurary With Minimum Cluster Size = 32'.format(
            acc)
        plt.title(title)
        plt.savefig('./plots/{}.jpg'.format(title))
        plt.clf()


if __name__ == '__main__':
    main()
