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

    from scipy.stats import kurtosis

    def get_kurtosis(input):
        kur = []
        for feature in range(input.shape[1]):
            kur.append(kurtosis(input[:,feature]))
        return kur

    my_kurtosis = get_kurtosis(X)
    print my_kurtosis
    print len(my_kurtosis)
    plt.scatter(range(1,17),my_kurtosis)
    plt.show()

