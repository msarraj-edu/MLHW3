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

    import numpy as np
    from scipy import linalg
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    from sklearn import mixture

    learning_algorithm = 'EM'

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


    with open(save_file_name, 'wb') as fp:
        a = csv.writer(fp, delimiter=',')

        a.writerow(["Dataset",
                    "Algorithm",
                    "Training Time",

                    "homo",
                    "compl",
                    "v-meas",
                    "ARI",
                    "AMI",
                    "silhouette",
                    "AIC",
                    "BIC"])


    train = pd.read_csv(file_path)
    print train.head()
    train.drop(drop_columns, axis=1, inplace=True)
    train.replace(to_replace='?', value=np.NaN, inplace=True)
    train = train.dropna(thresh=train.shape[1])
    X = train.drop([output_flag], axis=1)
    y = train[output_flag].values


    print("Number of NA values : {0}".format((X.shape[0] * X.shape[1]) - X.count().sum()))
    X = X.fillna(-1)



    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=0)

    start = time.time()

    parameters = {'init': ['kmeans', 'random'],
                  'n_clusters': range(2, len(le.classes_) + 1 + 5),
                  'n_init': [10]}

    estimator = KMeans()
    k_fold = 10
    cross_validation_parameter = None
    if is_cross:
        cross_validation_parameter = ShuffleSplit(X_train.shape[0], n_iter=k_fold, test_size=1. / k_fold,
                                                  random_state=1)
    # searcher = grid_search.GridSearchCV(estimator, parameters, cv=cross_validation_parameter, verbose=10)
    # searcher.fit(X_train, y_train)

    total_time = time.time() - start

    # save(filename=save_file_name)
    save("test2", globals())

    # print searcher.best_estimator_, searcher.best_score_
    # print total_time
    # print 'Finished ' + learning_algorithm

    lowest_bic = np.infty
    bic = []
    n_components_range = parameters['n_clusters']
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for init_parameter in parameters['init']:
        for cv_type in cv_types:
            for n_cluster in parameters['n_clusters']:
                bench_em(estimator=mixture.GaussianMixture(init_params=init_parameter, n_components=n_cluster,covariance_type=cv_type),
                              name=init_parameter + ' ' + cv_type + ' ' + str(n_cluster),
                              data=X,
                              save_file_name=save_file_name,
                              current_dataset=current_dataset)






