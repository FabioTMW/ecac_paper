import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import preprocessing 
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
from tabulate import tabulate

#nr_clusters = 50

#name of dataset, optimal number of clusters according tot he elbow rule
datasets = [
['balance-scale', 8],
#['bupa', 5],
#['glass', 7],
#['haberman', 5],
['ionosphere', 3],
['iris', 4],
#['mfeat', 4],
['new-thyroid', 6],
['optdigits', 50],
#['o-ring-erosion-only', 2],
#['o-ring-erosion-or-blowby', 3],
#['page-blocks', 5],
#['pima-indians-diabetes', 4],
#['post-operative', 5],
['quadrupeds', 3],
['segmentation', 5],
['sonar', 50],
['spambase', 10],
['text_class', 8],
['vehicle', 5],
['vowel', 30],
['waveform21', 3],
['waveform40', 4],
['wdbc', 4],
['wine', 4],
#['yeast', 6]
]


def analyse_datasets():
    

    table = []

    for i in range(len(datasets)):
        
        k = 0
        average_score_base_dt = 0
        average_score_cluster_dt = 0
        nr_runs = 5

        print('Running in dataset ', datasets[i][0])

        while k < nr_runs:
            alg_result = org_cluster(False, datasets[i][0], datasets[i][1])
            average_score_base_dt += alg_result[0]
            average_score_cluster_dt += alg_result[1]
            k += 1
        
        average_score_base_dt /= nr_runs
        average_score_cluster_dt /= nr_runs

        row = []
        row.append(datasets[i][0])
        row.append(datasets[i][1])
        row.append(average_score_cluster_dt)
        row.append(average_score_base_dt)
        table.append(row)

    print(tabulate(table, headers=["Dataset","Nr of Clusters", "DT By Cluster Acc", "Base DT Acc"], tablefmt="github"))
    



def org_cluster(visualize_clusters, dataset, nr_clusters):
    
    #tested datasets: 
    # vehicle, text_class, yeast, balance-scale, bupa, 
    # glass, haberman, ionosphere, mfeat, new-thyroid,
    # optdigits, o-ring-erosion-only , o-ring-erosion-or-blowby, page-blocks, pima-indians-diabetes
    # post-operative, quadrupeds, segmentation, sonar, spambase
    # vowel, waveform21, waveform40, wdbc, wine
    #partionated trees might not learn how to classify all labels since they never see them: lrs, ecoli
    
    path = "datasets/{}/{}.data".format(dataset, dataset)

    data = pandas.read_csv(path, header=None)

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
    clf = tree.DecisionTreeClassifier(random_state=42)
    clf.fit(x_train_var_baseline, x_train_res_baseline)
    baseline_score = clf.score(x_test_var, x_test_res)

    
    #baseline tree

    #instantiates kmeans algorithm with the number of clusters defined initially,
    #random_state is a random integer set so the algorithm runs the same way everytime
    kmeans = KMeans(n_clusters=nr_clusters, n_init=50, max_iter=1000, n_jobs=3, random_state=42)

    #train data without the target value
    x_train_no_target = x_train.iloc[:,0:len(data.columns)-1].values
    
    #print(preprocessing.normalize(x_train_no_target))
    
    #builds the clusters using the algorithm
    y_pred = kmeans.fit_predict(x_train_no_target)
    

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

    #print(data_by_cluster)

    j = 0
    decision_trees = []
    estimator = []
    weight = []
    
    #creates a tree for each cluster created
    while j < nr_clusters:
        clf = tree.DecisionTreeClassifier( random_state=42)
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
    eclf.fit(x_test_var, np.ravel(x_test_res))#does nothing as refit=false so it uses the dts previously built
    score = eclf.score(x_test_var, x_test_res)
    #print('Ensemble accuracy: ', score)

    test_data_by_cluster = [[] for Null in range(nr_clusters)]
    test_target_by_cluster = [[] for Null in range(nr_clusters)]
    
    #print(test_data_by_cluster)

    #determine what tree will be used by organizing test data by cluster
    for q in range(len(x_test_var)):
        
        points = np.append(kmeans.cluster_centers_, [x_test_var.values[q]], axis=0)
        #print(points)
        #gets the first 2 closest neighbors to his test data example using euclidean distance; first point is always the point itself
        knn = NearestNeighbors(n_neighbors=2)
        knn.fit(points)
        close_clusters = knn.kneighbors([points[nr_clusters]], return_distance=False)
        #print(close_clusters)
        #print(close_clusters[0][1])
        #print(np.asarray(x_test_var)[q])
        #assigns each test data example to an index; e.g. if the test data is in the first index, it will use the 1st tree model 
        test_data_by_cluster[close_clusters[0][1]].append(np.asarray(x_test_var)[q])
        test_target_by_cluster[close_clusters[0][1]].append(np.asarray(x_test_res)[q])

    
    total_acc = 0
    trees_used = 0
    for z in range(len(decision_trees)):
        score = 0
        if len(test_data_by_cluster[z]) != 0:
            #applied the specialized trees to the organized data
            #score = decision_trees[z].score(test_data_by_cluster[z], test_target_by_cluster[z])
            predicted = decision_trees[z].predict(test_data_by_cluster[z])
            acc_score = accuracy_score(test_target_by_cluster[z], predicted)
            trees_used += 1
            total_acc += acc_score
            #print('DT Nr', z+1)
            #print('Score:', acc_score)

        #total_acc += score

    total_acc = total_acc / trees_used

    #print("Baseline DT Score: ", baseline_score)
    #print('Decision Tree to Cluster Score:', total_acc)

    return [baseline_score, total_acc]


def elbow_method_datasets():
    i= 0
    while i < len(datasets):
        elbow_method(datasets[i][0])
        i += 1
    
    

def elbow_method(dataset):

    path = "datasets/{}/{}.data".format(dataset, dataset)

    data = pandas.read_csv(path, header=None)

    data.head()

    x = data

    x_train, x_test = train_test_split(x, test_size=0.2)

    x_train_no_target = x_train.iloc[:,0:len(data.columns)-1].values

    Error =[]

    for i in range(1, 50):
        kmeans = KMeans(n_clusters = i).fit(x_train_no_target)
        kmeans.fit(x_train_no_target)
        Error.append(kmeans.inertia_)
    
    plt.plot(range(1, 50), Error)
    plt.title(dataset)
    plt.xlabel('Nr clusters')
    plt.ylabel('Error')
    plt.show()



#elbow_method_datasets()
#elbow_method('haberman')
analyse_datasets()