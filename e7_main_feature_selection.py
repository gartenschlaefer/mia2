"""
mia2 - feature selection
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat

from mia2 import calc_dp, calc_fisher_ratio, feature_filter, feature_wrapper


if __name__ == "__main__":
    """
    main function of feature selection
    """

    # plot paths
    plot_path = 'ignore/ass7_data/plots/'

    # load data
    data = loadmat( './ignore/ass7_data/bspdrums.mat' )

    # get data arrays x:[n x m] n samples, m features
    x  = data[ 'drumFeatures' ][0][0][0]
    y  = data[ 'drumFeatures' ][0][0][1]

    # get shape of things
    n, m = x.shape

    # get labels
    labels = np.unique(y)

    # print some info
    print("num samples: {}, num features: {}, labels: {}".format(n, m, labels))


    # --
    # Goal: decrease feature space to low dimensions, which features are important 
    # Note: about 4-7 features should be left over
    # Two approaches (choose one or both):
    #   1) filter approach
    #   2) wrapper approach
    #
    # Are 3 features enough?

    # fisher ratio
    # r, l = calc_fisher_ratio(x, y)
    # print("fisher: ", r)
    # print("fisher: ", r.shape)
    # print("labels: ", l)

    # check discriminance potential
    print("\nfull set of features m={} has dp: {}".format(m, calc_dp(x, y)))


    # TODO: filter approach -> mia2 lib
    #x_filter, m_f, dp_m = feature_filter(x, y, algorithm='sfs')
    #x_filter, m_f, dp_m = feature_filter(x, y, algorithm='sbs')
    x_filter, m_f, dp_m = feature_filter(x, y, algorithm='lrs')

    # TODO: wrapper approach (Nico) -> mia2 lib
    x_wrapper, m_w = feature_wrapper(x, y)


    # calc discriminance potential
    dp_f = calc_dp(x_filter, y)
    dp_w = calc_dp(x_wrapper, y)

    # print selected features
    print("\nActual features by filter approach: \n", m_f)

    # some prints
    print("\nfeature filter m={} has dp: {}".format(len(m_f), dp_f))
    print("\nfeature wrapper m={} has dp: {}\n".format(len(m_w), dp_w))


    # plot dp
    plt.figure()
    plt.stem(dp_m, use_line_collection=True)
    plt.ylabel("dp")
    plt.xlabel("search iteration")
    plt.show()


   