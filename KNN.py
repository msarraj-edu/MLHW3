

if __name__ == "__main__":

    from sklearn import cross_validation
    import pandas as pd
    import numpy as np
    import time
    from sklearn.neighbors import KNeighborsClassifier



    learning_algorithm = 'KNN'

    print 'Starting '+learning_algorithm



    def save(filename='tmp',globals_=None,append=False):
       import shelve
       parameter = 'n'
       if append:
           parameter = 'c'
       globals_ = globals_ or globals()
       my_shelf=  shelve.open(filename,parameter)
       for key, value in globals_.items():
           if not key.startswith('__'):
               try:
                   my_shelf[key] = value
               except Exception:
                   print('ERROR shelving: "%s"' % key)
               else:
                   print('shelved: "%s"' % key)
       my_shelf.close()

    def load(filename='tmp',globals_=None):
       import shelve
       my_shelf = shelve.open(filename)
       for key in my_shelf:
           globals()[key]=my_shelf[key]
       my_shelf.close()



    current_dataset = 'letter'
    is_cross = True

    path_dict = {'letter':'../data/letter_recognition/letter-recognition.csv',
                 'ozone':'../data/ozone/onehr.csv'}

    class_dict = {'letter':['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X',
                         'Y','Z'],
                  'ozone':[0,1]}

    output_dict = {'letter':'\tlettr\tcapital','ozone':'Output'}
    drop_dict = {'letter':[],'ozone':['Date']}

    file_path = path_dict[current_dataset]
    output_flag = output_dict[current_dataset]
    class_names = class_dict[current_dataset]
    drop_columns = drop_dict[current_dataset]


    if is_cross:
        cross_str = 'cross_'
    else:
        cross_str = ''


    save_file_name = cross_str + {'letter':learning_algorithm + '_char','ozone':learning_algorithm + '_ozone'}[current_dataset]









    train = pd.read_csv(file_path)
    print train.head()
    train.drop(drop_columns,axis=1,inplace=True)
    train.replace(to_replace='?',value=np.NaN,inplace = True)
    train = train.dropna(thresh=train.shape[1])
    X = train.drop([output_flag], axis=1)
    y = train[output_flag].values


    print("Number of NA values : {0}".format((X.shape[0] * X.shape[1]) - X.count().sum()))
    X = X.fillna(-1)

    print X.head()
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=0)






    start = time.time()

    from sklearn import  grid_search
    from sklearn.cross_validation import ShuffleSplit





    parameters = {'n_neighbors':range(1,151),
                  'weights':['uniform','distance'],
                 'algorithm':['kd_tree'] ,
                 'leaf_size': [10,20,30,40,50] ,
                  'p': [2],
                 'metric':['minkowski']}






    estimator = KNeighborsClassifier()
    k_fold = 10
    cross_validation_parameter = None
    if is_cross:
        cross_validation_parameter = ShuffleSplit(X_train.shape[0],n_iter=k_fold,test_size=1./k_fold,random_state=1)
    searcher = grid_search.GridSearchCV(estimator,parameters,cv=cross_validation_parameter,verbose=10)
    searcher.fit(X_train,y_train)


    total_time = time.time() - start

    save(filename=save_file_name)



    print searcher.best_estimator_,searcher.best_score_
    print total_time
    print 'Finished ' + learning_algorithm

























