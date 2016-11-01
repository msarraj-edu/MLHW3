
import shelve
import time
from sklearn.neighbors import KNeighborsClassifier

ignore = ['load']
def load(filename='tmp',globals_=None):

   my_shelf = shelve.open(filename)
   for key in my_shelf:
        try:
            globals()[key]=my_shelf[key]
        except:
            continue
        #print key
   my_shelf.close()

main_directory = ''
data = 'char'
alg = 'KNN'
cross = ''
is_cross = True


if is_cross:
    cross = 'cross_'

data_file = main_directory + cross + alg +'_' + data
load(data_file)

print globals()
for key in dir():
    print key

print 'grid_scores_ ',searcher.grid_scores_
print 'best_estimator_  ',searcher.best_estimator_
print 'best_score_  ',searcher.best_score_
print 'best_params_   ',searcher.best_params_
print 'scorer_    ',searcher.scorer_
print X_train.shape

from matplotlib import pyplot as plt
from sklearn.learning_curve import learning_curve
import  numpy as np
from sklearn import  cross_validation


X_no_cross, X_cross, y_no_cross, y_cross = cross_validation.train_test_split(X_train, y_train, test_size=0.1, random_state=0)

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(0.01, 1.0, 15)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


from sklearn.ensemble import AdaBoostClassifier as ada
dataset_name = {'char':'Character ','ozone':'Ozone '}

if is_cross:
    prefix = alg+' with cross-validation\nLearning Curves for best parameters (Best C: '+ str(searcher.best_estimator_.n_neighbors)+ ',\n Best gamma: ' +str(searcher.best_estimator_.weights) + ' )'
else:
    prefix = alg + '\nLearning Curves for best parameters (Best K: ' + str(
        searcher.best_estimator_.n_neighbors) + ',\n Best weight: ' + str(
        searcher.best_estimator_.weights) + ' )'
prefix = dataset_name[data] + prefix
##############################################################################################

plot_learning_curve(KNeighborsClassifier(**searcher.best_params_),
                    prefix,
                    X_train,
                    y_train,
                    train_sizes=np.linspace(1.,0.2,num=50,endpoint=False),
                    cv=cross_validation_parameter
)


plt.savefig('Learning Curve ' + cross + alg +'_' + data)
plt.clf()
plt.close()




from matplotlib import  pyplot as plt
import  numpy as np
###############################################################################################################
config = dict(searcher.best_params_)


best_param_value = searcher.best_params_



changing_param = 'n_neighbors'
Ks = range(1,X_train.shape[1]+1)
print Ks
train_scores = np.zeros(len(Ks))
test_scores = np.zeros(len(Ks))

weights = ['distance']
for idx,weight in enumerate(weights):
    for jdx,k in enumerate(Ks):
        print k,time.strftime('%X %x %Z')
        config.update({changing_param:k})
        # clf = SVC(**config)
        clf = KNeighborsClassifier(**config)

        clf.fit(X_train,y_train)
        train_scores[jdx] = clf.score(X_train,y_train)
        test_scores[jdx] = clf.score(X_test,y_test)

# best_param_value = searcher.best_params_[changing_param]
# rest_of_title = {'n_neighbors':'(gamma= 1e-4)','weights':'=10)'}[changing_param]
fixed_param = {'n_neighbors':'weights','weights':'n_neighbors'}[changing_param]
rest_of_title = '(' + fixed_param + '= ' + str(best_param_value[fixed_param]) + ')'

# plt.figure()
plt.plot(Ks,test_scores[:], linewidth=3, label = "Test Score"+rest_of_title)
plt.plot(Ks,train_scores[:], linewidth=3, label = "Train Score"+rest_of_title)
# plt.xlim(Ks.min(),Ks.max())
plt.legend()
# plt.ylim(0.5, 1.01)
plt.xlabel(changing_param)
plt.ylabel("Score")
plt.title("Score vs. " + changing_param)
plt.savefig('score vs ' +changing_param)
plt.clf()
plt.close()


config = dict(searcher.best_params_)


# ################################################################################################################
#
from sklearn.metrics import classification_report, confusion_matrix
#
# #################################################Report############################################
#
gridScores = searcher.grid_scores_

clf = KNeighborsClassifier(**searcher.best_params_)
training_start = time.time()
clf.fit(X_train,y_train)
training_end = time.time()
testing_start = time.time()
y_pred = clf.predict(X_test)
testing_end = time.time()

training_time = training_end-training_start
testing_time = testing_end-testing_start

target_names = {'char':['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'],'ozone':['0','1']}[data]


st = "\nClassification report for classifier %s:\n%s\n" % (clf, classification_report(y_test, y_pred, target_names=target_names)) + '\n'
confusionMatrix =confusion_matrix(y_test, y_pred)
st = st + "Confusion matrix:\n%s" % confusionMatrix + '\n'
st = st + 'Training Time:' + str(training_time) + '\nTesting Time: ' + str(testing_time)

print st

report_file_name = 'report_'+data+alg+cross+'.txt'

tmpora = [1 if i == j else 0 for i,j in zip(y_pred,y_test) ]
print 'score: ',float(sum(tmpora))/len(tmpora)


f = open(report_file_name, 'w')
f.write(st)















