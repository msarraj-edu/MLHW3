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

    learning_algorithm = 'PCA'

    print 'Starting ' + learning_algorithm

    current_dataset = 'cancer'
    is_cross = True

    path_dict = {'letter': './letter-recognition.csv',
                 'ozone': '../data/ozone/onehr.csv',
                 "cancer":'./wdbc.csv'}

    class_dict = {
        'letter': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                   'U', 'V', 'W', 'X',
                   'Y', 'Z'],
        'ozone': [0, 1],
        'cancer':[0,1]}

    output_dict = {'letter': '1', 'ozone': 'Output','cancer':"2"}
    drop_dict = {'letter': [], 'ozone': ['Date'],'cancer':['1']}





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


    header_dict = {'letter': map(str,range(1,18)),
                   'cancer': map(str,range(1,33))}


    train = pd.read_csv(file_path,names=header_dict[current_dataset])
    print train.head()
    train.drop(drop_columns, axis=1, inplace=True)
    train.replace(to_replace='?', value=np.NaN, inplace=True)
    train = train.dropna(thresh=train.shape[1])
    X = train.drop([output_flag], axis=1)
    y = train[output_flag].values

    X = preprocessing.normalize(X,axis=0)

    # print("Number of NA values : {0}".format((X.shape[0] * X.shape[1]) - X.count().sum()))
    # X = X.fillna(-1)

    with open(current_dataset+' RP.csv', 'wb') as fp:
        a = csv.writer(fp, delimiter=',')

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


    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=0)

    start = time.time()

    parameters = {'init': ['kmeans', 'random'],
                  'n_clusters': range(2, len(le.classes_) + 1 + 5),
                  'n_init': [10]}

    estimator = KMeans()

    n_components = np.arange(0, 50, 1)  # options for n_components

    lowest_bic = np.infty
    bic = []
    # n_components_range = parameters['n_clusters']
    # cv_types = ['spherical', 'tied', 'diag', 'full']
    # for init_parameter in parameters['init']:
    #     for cv_type in cv_types:
    #         for n_cluster in parameters['n_clusters']:
    #             bench_em(estimator=mixture.GMM(init_params=init_parameter, n_components=n_cluster,covariance_type=cv_type),
    #                           name=init_parameter + ' ' + cv_type + ' ' + str(n_cluster),
    #                           data=X_train,
    #                           labels=y_train,
    #                           save_file_name=save_file_name,
    #                           current_dataset=current_dataset)

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import linalg

    from sklearn.decomposition import PCA, FactorAnalysis, FastICA
    from sklearn.random_projection import GaussianRandomProjection
    from sklearn.covariance import ShrunkCovariance, LedoitWolf
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import GridSearchCV
    from scipy.stats import kurtosis

    print(__doc__)


    def get_kurtosis(X):
        kur = []
        for feature in range(X.shape[1]):
            kur.append(kurtosis(X[:, feature]))

        return kur

    def save_to_csv(file,data,description, append=False):

        if append:
            with open(file, 'ab') as fp:
                a = csv.writer(fp, delimiter=',')
                myList = [description]
                myList.extend(data)
                a.writerow(myList)
        else:
            with open(file, 'wb') as fp:
                a = csv.writer(fp, delimiter=',')
                myList = [description]
                myList.extend(data)
                a.writerow(myList)


    def top_scorer(n):
        def my_func(estimator,X,y=None):
            kur = []
            for feature in range(X.shape[1]):
                kur.append(kurtosis(input[:, feature]))
            # sorted_kur = sorted(range(len(a)), key=lambda i: a[i], reverse=True)[:n]
            # avg = sum(sorted_kur)/len(sorted_kur)
            # result = []
            # for value in sorted_kur:
            #     if value >= avg:
            #         result.append(value)
            #
            # return result
            return kur

        return my_func


    def best_scorer(estimator, X, y=None):
        return 0.

    n_samples, n_features, rank = 1000, len(X[0]), 10


    n_components = np.arange(1, n_features+1, 1)  # options for n_components
    # n_components = np.arange(1,4)


    def save_data(estimator,X,file):
        estimator.fit(X)
        transformed_data = estimator.transform(X)

        headers = ['feature_' + str(i) for i in range(transformed_data.shape[1])]
        # np.savetxt(file,transformed_data,header=','.join(headers),delimiter=',')
        np.savetxt(file, transformed_data, delimiter=',')



    def compute_scores(X):
        pca = PCA(svd_solver='full')
        fa = FactorAnalysis(max_iter=3000)
        rp = GaussianRandomProjection()
        ica = FastICA()

        timings = np.zeros((n_components.size,5))

        pca_scores, fa_scores, rp_scores, ica_scores = [], [], [], []
        for idx,n in enumerate(n_components):
            print n,"components"
            pca.n_components = n
            fa.n_components = n
            rp.n_components = n
            ica.n_components = n
            timings[idx,0] = n


            start = time.time()
            pca_scores.append(np.mean(cross_val_score(pca, X)))
            timings[idx,1]=time.time()-start

            start = time.time()
            fa_scores.append(np.mean(cross_val_score(fa, X)))
            timings[idx, 2] = time.time() - start
            save_data(fa, X, current_dataset + '_FA_' + str(n) + '.csv')

            start = time.time()
            rp.fit(X)
            timings[idx, 3] = time.time() - start
            rp_data = rp.transform(X)
            save_data(rp,X,current_dataset+'_RP_'+ str(n) + '.csv')
            save_to_csv(current_dataset + ' RP.csv',get_kurtosis(rp_data),'rp for '+current_dataset+', k= %d'%n,append=True)
            if n == n_components[-1]:
                start = time.time()
                ica.fit(X)
                timings[idx, 4] = time.time() - start

                ica_data = ica.transform(X)
                save_to_csv(current_dataset+' ICA.csv',get_kurtosis(ica_data),'rp for '+current_dataset+', k= %d'%n)

        np.savetxt(current_dataset +' timings.csv',timings,delimiter=',',header='components,pca,fa,rp,ica')

        pca = PCA(svd_solver= 'full',n_components=n_components.max())
        save_data(pca,X,current_dataset+'_PCA.csv')
        ica = FastICA(n_components=n_components.max())
        save_data(ica,X,current_dataset+'_ICA.csv')


        save_to_csv(current_dataset + ' PCA.csv',pca_scores,'pca for ' + current_dataset)
        save_to_csv(current_dataset + ' FA.csv', fa_scores, 'fa for ' + current_dataset)

        return pca_scores, fa_scores


    def shrunk_cov_score(X):
        shrinkages = np.logspace(-2, 0, 30)
        cv = GridSearchCV(ShrunkCovariance(), {'shrinkage': shrinkages})
        return np.mean(cross_val_score(cv.fit(X).best_estimator_, X))


    def lw_score(X):
        return np.mean(cross_val_score(LedoitWolf(), X))


    title = current_dataset
    pca_scores, fa_scores = compute_scores(X)
    n_components_pca = n_components[np.argmax(pca_scores)]
    n_components_fa = n_components[np.argmax(fa_scores)]
    # n_components_rp = n_components[np.argmax(rp_scores)]
    # n_components_ica = n_components[np.argmax(ica_scores)]

    pca = PCA(svd_solver='full', n_components='mle')
    pca.fit(X)
    n_components_pca_mle = pca.n_components_

    print("best n_components by PCA CV = %d" % n_components_pca)
    print("best n_components by FactorAnalysis CV = %d" % n_components_fa)
    # print("best n_components by Random Projection CV = %d" % n_components_rp)
    # print("best n_components by ICA CV = %d" % n_components_ica)
    print("best n_components by PCA MLE = %d" % n_components_pca_mle)

    plt.figure()
    plt.plot(n_components, pca_scores, 'b', label='PCA scores')
    plt.plot(n_components, fa_scores, 'r', label='FA scores')
    # plt.plot(n_components, rp_scores, 'c', label='Random Projection scores')
    # plt.plot(n_components, ica_scores, 'g', label='ICA scores')
    plt.axvline(rank, color='g', label='TRUTH: %d' % rank, linestyle='-')
    plt.axvline(n_components_pca, color='b',
                label='PCA CV: %d' % n_components_pca, linestyle='--')
    plt.axvline(n_components_fa, color='r',
                label='FactorAnalysis CV: %d' % n_components_fa,
                linestyle='--')
    # plt.axvline(n_components_rp, color='c',
    #             label='Random Projection CV: %d' % n_components_rp,
    #             linestyle='--')
    # plt.axvline(n_components_ica, color='g',
    #             label='ICA CV: %d' % n_components_ica,
    #             linestyle='--')
    plt.axvline(n_components_pca_mle, color='k',
                label='PCA MLE: %d' % n_components_pca_mle, linestyle='--')

    # compare with other covariance estimators
    plt.axhline(shrunk_cov_score(X), color='violet',
                label='Shrunk Covariance MLE', linestyle='-.')
    plt.axhline(lw_score(X), color='orange',
                label='LedoitWolf MLE' % n_components_pca_mle, linestyle='-.')

    plt.xlabel('nb of components')
    plt.ylabel('CV scores')
    plt.legend(loc='lower right')
    plt.title(title)

    plt.show()






