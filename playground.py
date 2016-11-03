

import KNN
from time import time
from sklearn import  metrics
import csv

def print_dict(d):
    for key,val in d.iteritems():
        print key,val

def save(filename='tmp', globals_=None, append=False):
    import shelve
    parameter = 'n'
    if append:
        parameter = 'c'
    globals_ = globals_ or globals()
    my_shelf = shelve.open(filename, parameter)
    for key, value in globals_.items():
        tmp = value
        if not (key.startswith('__') or key in ["load","save","print_dict"] or callable(tmp)):
            try:
                my_shelf[key] = value
            except Exception:
                print('ERROR shelving: "%s"' % key)
            else:
                print('shelved: "%s"' % key)
    my_shelf.close()


def load(filename='tmp', globals_=None):
    import shelve
    my_shelf = shelve.open(filename)
    print my_shelf
    return my_shelf
    #
    # for key in my_shelf:
    #     if not (key.startswith("_") or key in ["load","save"]):
    #         # globals()[key] = my_shelf[key]
    #         print key,my_shelf[key]
    #         new_dic[key] = my_shelf[key]
    # my_shelf.close()


def local_func():
    a = 1
    b = 2
    c = {a:b}
    save(filename="test",globals_= locals())



def my_score(data,labels,centers,scoring_metric):

    sum = 0

    for point,label in zip(data,labels):
        sum+= scoring_metric(point,centers[label])
    return sum


def euc_distance(point1,point2):
    distance = 0
    for i,j in zip(point1,point2):
        distance += (i-j)*(i-j)
    return distance


def bench_k_means(estimator, name, data,save_file_name,current_dataset):
    t0 = time()
    estimator.fit(data)
    labels = estimator.labels_
    training_time = time() - t0
    homo = metrics.homogeneity_score(labels, estimator.labels_)
    complete = metrics.completeness_score(labels, estimator.labels_)
    v_score = metrics.v_measure_score(labels, estimator.labels_)
    ARI = metrics.adjusted_rand_score(labels, estimator.labels_)
    AMI = metrics.adjusted_mutual_info_score(labels,  estimator.labels_)
    silhouette=  metrics.silhouette_score(data, estimator.labels_,
                             metric='euclidean',
                             sample_size=1000)
    silhouette2 = metrics.silhouette_score(data, estimator.labels_,
                                          metric='euclidean',
                                          sample_size=1000)
    silhouette3 = metrics.silhouette_score(data, estimator.labels_,
                                          metric='euclidean',
                                          sample_size=1000)
    silhouette4 = metrics.silhouette_score(data, estimator.labels_,
                                          metric='euclidean',
                                          sample_size=1000)

    custom_score = my_score(data,labels,estimator.cluster_centers_,euc_distance)
    # bic = estimator.bic(data)
    # aic = estimator.aic(data)


    print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f   %.3f   %.3f   %.3f   %.3f   %.3f'
          % (name,
             (training_time),
             estimator.inertia_,
             homo,
             complete,
             v_score,
             ARI,
             AMI,
             silhouette,silhouette2,silhouette3,silhouette4,
             custom_score
             ))
    with open(save_file_name, 'ab') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerow([current_dataset,
                    name,
                    (training_time),
                    estimator.inertia_,
                    homo,
                    complete,
                    v_score,
                    ARI,
                    AMI,
                    silhouette,
                    custom_score
                    ])

def bench_em(estimator, name, data,save_file_name,current_dataset):
    t0 = time()
    estimator.fit(data)
    labels = estimator.predict(data)
    training_time = time() - t0
    labels_ = estimator.predict(data)
    homo = metrics.homogeneity_score(labels, labels_)
    complete = metrics.completeness_score(labels, labels_)
    v_score = metrics.v_measure_score(labels, labels_)
    ARI = metrics.adjusted_rand_score(labels, labels_)
    AMI = metrics.adjusted_mutual_info_score(labels,  labels_)
    silhouette=  metrics.silhouette_score(data, labels_,
                             metric='euclidean',
                             sample_size=None)
    bic = estimator.bic(data)
    aic = estimator.aic(data)


    print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f   %.3f   %.3f'
          % (name,
             (training_time),

             homo,
             complete,
             v_score,
             ARI,
             AMI,
             silhouette,
             aic,
             bic
             ))
    with open(save_file_name, 'ab') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerow([current_dataset,
                    name,
                    (training_time),

                    homo,
                    complete,
                    v_score,
                    ARI,
                    AMI,
                    silhouette,
                    aic,
                    bic
                    ])

if __name__ == "__main__":
    local_func()
    print globals()
    print "\n\n\n\n\n\n\n"

    load("test")
    d = dict(globals())
    for key,value in d.iteritems():
        print key,value

