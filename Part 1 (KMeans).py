import csv

if  __name__ == "__main__":



    from playground import load, save, print_dict, bench_k_means, my_score
    import pandas as pd
    import numpy as np
    import time
    from sklearn.cluster import KMeans
    from sklearn import preprocessing


    ############# Initializing Params ######################
    learning_algorithm = 'K Clustering'



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


    ################### Preprocessing Data ########################

    print 'Starting ' + learning_algorithm




    with open(save_file_name, 'wb') as fp:
        a = csv.writer(fp, delimiter=',')

        a.writerow(["Dataset",
                    "Algorithm",
                    "Training Time",
                    "inertia",
                    "homo",
                    "compl",
                    "v-meas",
                    "ARI",
                    "AMI",
                    "silhouette"])


    train = pd.read_csv(file_path)
    print train.head()
    train.drop(drop_columns, axis=1, inplace=True)
    train.replace(to_replace='?', value=np.NaN, inplace=True)
    train = train.dropna(thresh=train.shape[1])
    X = train.drop([output_flag], axis=1)
    y = train[output_flag].values

    X = preprocessing.normalize(X, axis=0)




    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)



    ################### Algorithm Computation #############################

    start = time.time()

    parameters = {'init': ['k-means++','random'],
                  'n_clusters': range(2,len(le.classes_)+1 + 5),
                  'n_init': [10]}


    parameters["n_clusters"] = range(25,28)


    estimator = KMeans()

    for init_parameter in parameters['init']:
        for n_cluster in parameters['n_clusters']:
            bench_k_means(estimator= KMeans(init=init_parameter,   n_clusters=n_cluster),
                          name= init_parameter+' '+str(n_cluster),
                          data= X,
                          save_file_name= save_file_name,
                          current_dataset= current_dataset
                          )

    print "Computation Time: ",time.time()-start










