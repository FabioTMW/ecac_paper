import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import EnsembleVoteClassifier

nr_clusters = 5 

def org_cluster(check_optimal_cluster_nr, visualize_clusters):
    
    #tested datasets: text_class, iris, yeast
    data = pandas.read_csv('datasets/iris/iris.data', header=None)

    data.head()

    x = data

    #train/test data split
    x_train, x_test = train_test_split(x, test_size=0.2)

    #baseline tree data
    x_train_var_baseline = x_train.iloc[:,0:len(data.columns)-1]
    x_train_res_baseline = x_train.iloc[:,len(data.columns)-1:len(data.columns)]
    #baseline tree data

    x_test_var = x_test.iloc[:,0:len(data.columns)-1]
    x_test_res = x_test.iloc[:,len(data.columns)-1:len(data.columns)]

    #baseline tree
    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train_var_baseline, x_train_res_baseline)
    score = clf.score(x_test_var, x_test_res)

    print("Global DT Pred Score: ", score)
    #baseline tree

    #instantiates kmeans algorithm with the number of clusters defined initially,
    #random_state is a random integer set so the algorithm runs the same way everytime
    kmeans = KMeans(n_clusters=nr_clusters, n_init=50, max_iter=1000, n_jobs=3)

    #train data without the target value
    x_train_no_target = x_train.iloc[:,0:len(data.columns)-1].values
    
    #builds the clusters using the algorithm
    y_pred = kmeans.fit_predict(x_train_no_target)
    
    #optional: elbow method to check the optimal cluster number
    if check_optimal_cluster_nr:
        elbow_method(x_train_no_target,12)
        

    #optional: visualize output for the cluster, they end up overlapping as long as there are more than 3 features
    if visualize_clusters:
        plt.scatter(x_train_no_target[:, 0], x_train_no_target[:, 1], c=y_pred)
        plt.show()

    
    data_by_cluster = [[] for Null in range(nr_clusters)]
    target_by_cluster = [[] for Null in range(nr_clusters)]

    x_train_target = x_train.iloc[:,len(data.columns)-1:len(data.columns)].values
    
    #organize data into features and target classes by cluster number
    i = 0
    while i < len(x_train):
        data_by_cluster[y_pred[i]].append(x_train_no_target[i].tolist())
        target_by_cluster[y_pred[i]].append(x_train_target[i].tolist())
        i += 1

    j = 0
    decision_trees = []
    estimator = []
    weight = []
    
    #creates a tree for each cluster created
    while j < nr_clusters:
        clf = tree.DecisionTreeClassifier()
        #uses train data contained in the cluster to creatre a tree 
        
        clf.fit(np.asarray(data_by_cluster[j]), np.asarray(target_by_cluster[j]))
        decision_trees.append(clf)

        #voting ensemble components
        estimator.append(clf)
        weight.append(1)#same weight for each tree
        
        j += 1

    #each tree doing classification in all test entries as test

    #for z in range(len(decision_trees)):
    #  score = decision_trees[z].score(x_test_var, x_test_res)
    #  print('DT Nr', z+1)
    #  print('Score:', score)

    #Majority voting classifier using the trees built
    eclf = EnsembleVoteClassifier(clfs=estimator, weights=weight, refit=False)
    eclf.fit(x_test_var, np.ravel(x_test_res))#does nothing as refit=false so it uses the dt previously built
    score = eclf.score(x_test_var, x_test_res)

    print('Ensemble accuracy: ', score)
    

def elbow_method(pdata, nr_clusters):
    Error =[]
    for i in range(1, nr_clusters):
        kmeans = KMeans(n_clusters = i).fit(pdata)
        kmeans.fit(pdata)
        Error.append(kmeans.inertia_)
    import matplotlib.pyplot as plt
    plt.plot(range(1, nr_clusters), Error)
    plt.title('Elbow method')
    plt.xlabel('Nr clusters')
    plt.ylabel('Error')
    plt.show()


#org_cluster(True, True)
#org_cluster(False, True)
#org_cluster(True, False)
org_cluster(False, False)