import csv

if  __name__ == "__main__":

    # saving
    # save("test2",globals())
    #
    #
    # loading
    # new_vars = load("test2")
    # globals().update(new_vars)

    from playground import load, save, print_dict, bench_em
    from sklearn import cross_validation
    import pandas as pd
    import numpy as np
    import time
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn import grid_search
    from sklearn.cross_validation import ShuffleSplit
    from sklearn.cluster import KMeans
    from sklearn import preprocessing
    import itertools

    import warnings

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import linalg

    from sklearn.linear_model import (RandomizedLasso, lasso_stability_path,
                                      LassoLarsCV)
    from sklearn.feature_selection import f_regression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import auc, precision_recall_curve
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.utils.extmath import pinvh
    # from sklearn.exceptions import ConvergenceWarning

    import numpy as np
    from scipy import linalg
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    from sklearn import mixture

    learning_algorithm = 'PCA'

    print 'Starting ' + learning_algorithm

    current_dataset = 'letter'
    is_cross = True

    path_dict = {'letter': './letter-recognition.csv',
                 'ozone': '../data/ozone/onehr.csv',
                 "cancer":'./Cancer.csv'}

    class_dict = {
        'letter': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                   'U', 'V', 'W', 'X',
                   'Y', 'Z'],
        'ozone': [0, 1],
        'cancer':[0,1]}

    output_dict = {'letter': '\tlettr\tcapital', 'ozone': 'Output','cancer':"Output"}
    drop_dict = {'letter': [], 'ozone': ['Date'],'cancer':[]}





    file_path = path_dict[current_dataset]
    output_flag = output_dict[current_dataset]
    class_names = class_dict[current_dataset]
    drop_columns = drop_dict[current_dataset]

    if is_cross:
        cross_str = 'cross_'
    else:
        cross_str = ''

    save_file_name = cross_str + {'letter': learning_algorithm + '_letter',
                                  'ozone': learning_algorithm + '_ozone',
                                  'cancer':learning_algorithm + '_cancer'}[current_dataset] +'.csv'


    # with open(save_file_name, 'wb') as fp:
    #     a = csv.writer(fp, delimiter=',')
    #
    #     a.writerow(["Dataset",
    #                 "Algorithm",
    #                 "Training Time",
    #
    #                 "homo",
    #                 "compl",
    #                 "v-meas",
    #                 "ARI",
    #                 "AMI",
    #                 "silhouette",
    #                 "AIC",
    #                 "BIC"])


    train = pd.read_csv(file_path)
    print train.head()
    train.drop(drop_columns, axis=1, inplace=True)
    train.replace(to_replace='?', value=np.NaN, inplace=True)
    train = train.dropna(thresh=train.shape[1])
    X = train.drop([output_flag], axis=1)
    y = train[output_flag].values

    X = preprocessing.normalize(X,axis=0)

    # print("Number of NA values : {0}".format((X.shape[0] * X.shape[1]) - X.count().sum()))
    # X = X.fillna(-1)




    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=0)

    n_features = X.shape[1]

    coef_min = 0.2
    coef = np.zeros(n_features)
    coef[:]= coef_min

    ###########################################################################
    # Plot stability selection path, using a high eps for early stopping
    # of the path, to save computation time
    alpha_grid, scores_path = lasso_stability_path(X, y, random_state=42,
                                                   eps=0.05)
    # print alpha_grid
    # print scores_path

    for i in range(0,scores_path.shape[0]):
        if scores_path[i,0] > 0.1:
            print i+1
    plt.figure()
    # We plot the path as a function of alpha/alpha_max to the power 1/3: the
    # power 1/3 scales the path less brutally than the log, and enables to
    # see the progression along the path
    hg = plt.plot(alpha_grid[1:] ** .333, scores_path[coef != 0].T[1:], 'r')
    # hb = plt.plot(alpha_grid[1:] ** .333, scores_path[coef == 0].T[1:], 'k')
    hb = plt.plot(alpha_grid[1:] ** .333, scores_path.T[1:], 'r')

    ymin, ymax = plt.ylim()
    plt.xlabel(r'$(\alpha / \alpha_{max})^{1/3}$')
    plt.ylabel('Stability score: proportion of times selected')
    # plt.title('Stability Scores Path - Mutual incoherence: %.1f' % mi)
    plt.axis('tight')
    plt.legend((hg[0], hb[0]), ('relevant features', 'irrelevant features'),
               loc='best')

    ###########################################################################
    # Plot the estimated stability scores for a given alpha

    # Use 6-fold cross-validation rather than the default 3-fold: it leads to
    # a better choice of alpha:
    # Stop the user warnings outputs- they are not necessary for the example
    # as it is specifically set up to be challenging.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        # warnings.simplefilter('ignore', ConvergenceWarning)
        lars_cv = LassoLarsCV(cv=6).fit(X, y)

    # Run the RandomizedLasso: we use a paths going down to .1*alpha_max
    # to avoid exploring the regime in which very noisy variables enter
    # the model
    alphas = np.linspace(lars_cv.alphas_[0], .1 * lars_cv.alphas_[0], 6)
    clf = RandomizedLasso(alpha=alphas, random_state=42).fit(X, y)
    trees = ExtraTreesRegressor(100).fit(X, y)
    # Compare with F-score
    F, _ = f_regression(X, y)

    plt.figure()
    for name, score in [('F-test', F),
                        ('Stability selection', clf.scores_),
                        ('Lasso coefs', np.abs(lars_cv.coef_)),
                        ('Trees', trees.feature_importances_),
                        ]:
        precision, recall, thresholds = precision_recall_curve(coef != 0,
                                                               score)
        plt.semilogy(np.maximum(score / np.max(score), 1e-4),
                     label="%s. AUC: %.3f" % (name, auc(recall, precision)))

    # plt.plot(np.where(coef != 0)[0], [2e-4] * n_relevant_features, 'mo',
    #          label="Ground truth")
    # plt.xlabel("Features")
    # plt.ylabel("Score")
    # # Plot only the 100 first coefficients
    # plt.xlim(0, 100)
    # plt.legend(loc='best')
    # plt.title('Feature selection scores - Mutual incoherence: %.1f'
    #           % mi)

plt.show()


