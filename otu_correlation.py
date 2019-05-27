from sklearn.linear_model import LassoCV
from sklearn.datasets import make_regression
from external_dataset import load_biogas

from sklearn.linear_model import LassoLarsCV
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

try:
    import numpy as np
except:
    import _numpypy as np

from sklearn.svm import LinearSVR
from sklearn import linear_model

from itertools import combinations


from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer, fetch_lfw_people

import matplotlib.pyplot as plt
from collections import defaultdict

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

import sys

import json

from tqdm import tqdm

import argparse

from multiprocessing import Pool
import multiprocessing

from scipy import stats

def plot_samples(dataX,dataY,labels):
    plt.subplot()
    plt.title("")
    plt.scatter(dataX, dataY, marker='o', c=labels,
                s=50, edgecolor='k')
    plt.show()

def binary_intersect_size(class1, class2):
    min1 = min(class1)
    min2 = min(class2)
    max1 = max(class1)
    max2 = max(class2)
    
    if(max2 > max1):
        return min2-max1
    else:
        return min1-max2

def multiclass_intersect_size(X,y, number_classes):
    heuristic_sum = 0
    for i,j in combinations(range(number_classes),2):
        relative_dis = binary_intersect_size(X[y==i],X[y==j])
        # relative_dis = 0 if (relative_dis > 0) else relative_dis  
        # relative_dis = -1 if (relative_dis == 0) else relative_dis  
        heuristic_sum += relative_dis
    return heuristic_sum


def dim_instance_separability_binary(X, y):
    class1 = X[y==0]
    class2 = X[y==1]
    min1 = min(class1)
    min2 = min(class2)
    max1 = max(class1)
    max2 = max(class2)
    
    if(max2 > max1):
        center = (max1+min2)/2
        min_dir = min2
        max_esq = max1
        dir_class = 1
        esq_class = 0
    else:
        center = (max2+min1)/2
        min_dir = min1
        max_esq = max2
        dir_class = 0
        esq_class = 1

    instances_covered = []
    for i,x in enumerate(X):
        if(x > center and y[i]==dir_class) or (x < center and y[i]==esq_class):
            instances_covered.append(i)

    
    # print(X,y,min_dir,max_esq)
    return set(instances_covered)

def dim_instance_separability(X,y):
    instances_covered = set()
    for i in range(len(set(y))):
        copy_y = np.array(y)
        copy_y[y!=i] = 0
        copy_y[y==i] = 1
        for inst in dim_instance_separability_binary(X,copy_y):
            if(copy_y[inst] == 1):
                instances_covered = instances_covered.union({inst})
    # if(len(set(instances_covered))>=15):
        # plot_samples(X, X, y)
    return set(instances_covered)


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Transform the result from SparCC, Kendall, Spearman and Pearson (.out files) as graph into json files.')
    parser.add_argument('--dtype', action = 'store', dest = 'dtype',
                                default = 'all_relative', required = True,
                                help = 'The type of spreedsheet used. See external_dataset.py for more information')

    parser.add_argument('--json_name', action = 'store', dest = 'json_name',
                                    default = None, required = True,
                                    help = 'The name of json file')

    parser.add_argument('--taxa_level', action = 'store', dest = 'taxa_level',
                                    default = -1, required = False, type=int,
                                    help = 'The index of taxa level column indicate in spredsheet. See external_dataset.py for more information')

    parser.add_argument('--label_type', action = 'store', dest = 'label_type',
                                    default = "grouped", required = False,
                                    help = 'See external_dataset.py for more information')

    parser.add_argument('--abundance_limiar', action = 'store', dest = 'abundance_limiar',
                                    default = 1.0, required = False, type=float,
                                    help = 'See external_dataset.py for more information')
    
    parser.add_argument('--filter_by_taxa_name', action = 'store', dest = 'filter_by_taxa_name',
                                    default = None, required = False,
                                    help = 'See external_dataset.py for more information')
    
    parser.add_argument('--score_type', action = 'store', dest = 'score_type',
                                    default = "reg", required = False,
                                    choices=['reg', 'pearson','spearman','kendall'],
                                    help = 'Evaluation used to compute correlation between OTU.')
    parser.add_argument('--filter_by_taxa_level', action = 'store', dest = 'filter_by_taxa_level',
                                    default = None, required = False, type=int,
                                    help = 'See external_dataset.py for more information')
    parser.add_argument('--limiar_alpha', action = 'store', dest = 'LIMIAR_ALPHA',
                                    default = None, required = False, type=float,
                                    help = 'The limiar used to filter the network correlation edges. Represent the number of samples repetitions that is necessary to consider a correlation between two differents OTU')
    parser.add_argument('--limiar_beta', action = 'store', dest = 'LIMIAR_BETA',
                                    default = None, required = False, type=float,
                                    help = 'The limiar used to filter the network correlation edges. Represent the relative abundance necessary to consider a sample as active.')
    parser.add_argument('--workers', action = 'store', dest = 'workers',
                                    default = None, required = False, type=int,
                                    help = 'Number of workers used to parallelize processing.')
    
    arguments = parser.parse_args()
    cores = arguments.workers
    if cores is None:
        cores = multiprocessing.cpu_count()

    
    
    score_type = arguments.score_type
    taxa_level = arguments.taxa_level
    label_type = arguments.label_type
    abundance_limiar = arguments.abundance_limiar
    LIMIAR_ALPHA = arguments.LIMIAR_ALPHA
    LIMIAR_BETA = arguments.LIMIAR_BETA

    filter_by_taxa_level = None
    if not (arguments.filter_by_taxa_name is None) and not (arguments.filter_by_taxa_level is None):
        filter_by_taxa_level = (arguments.filter_by_taxa_level, arguments.filter_by_taxa_name)
    
    dtype = arguments.dtype
    JSON_NAME = arguments.json_name

    db = load_biogas(data_type=dtype, label_type=label_type, label_value='name',
                     relative_taxa_level=taxa_level, abundance_limiar=abundance_limiar,
                     filter_by_taxa_level=filter_by_taxa_level)
    X, dataY = (db.data, db.target)

    # Normalize by column
    # for i in range(X.shape[1]):
    #     X[:,i] = X[:,i]/max(X[:,i])

    all_classes = set(dataY)
    number_classes = len(all_classes)
    total = set()

    edges = []

    dims_cover = []
    for i in range(X.shape[1]):
        dim_cover = dim_instance_separability(
            X[:,i],
            dataY
        )
        dim_intersect = multiclass_intersect_size(
            X[:,i],dataY, number_classes
        )

        dims_cover.append(
            [i,dim_cover,dim_intersect]
        )
        total = total.union(dim_instance_separability(X[:,i],dataY))

    max_sample_separation = len(set(total))


    dims_cover = sorted(dims_cover, key=lambda dim: dim[2], reverse=True)


    
    CO_OCORRENCIE_LIMIAR = LIMIAR_BETA
    if CO_OCORRENCIE_LIMIAR is None:
        CO_OCORRENCIE_LIMIAR = 0.8
    
    N_COMB = 2
    # LIMIAR = 1
    # LIMIAR = 0.5
    # LIMIAR = 0.37
    # LIMIAR = 0.137
    # LIMIAR = 0.99
    # LIMIAR = 0.01

    
    
    
    INT_ORGS = set()
    PLOT = False
    dims_cover = np.array(dims_cover)
    GRAPH_ORGS = defaultdict(list)
    
    # score_type = "pearson"
    # score_type = "reg"
    # score_type = "coc"
    # score_type = "dt"
    # score_type = "cover"

    total_combinations = combinations(range(len(dims_cover)),N_COMB)
    if score_type == "dt":
        pass
    elif score_type == "cover":
        pass
    elif score_type == "coc":
        if LIMIAR_ALPHA is None:
            LIMIAR = 2.0/X.shape[1]
        else:
            LIMIAR = float(LIMIAR_ALPHA)/X.shape[0]
    elif score_type=="reg" or score_type=="pearson" or score_type=="spearman" or score_type=="kendall":
        if LIMIAR_ALPHA is None:
            LIMIAR = 0.95
        else:
            LIMIAR = float(LIMIAR_ALPHA)
            
    print("Testing to",db.name," with",N_COMB,"combinations and limiar equal to",LIMIAR, "with",score_type," score type")
    print("Max cover "+str(max_sample_separation)+" from "+str(len(X)))

    def count_combinations(n,k):
        total = 1
        for i in range(k):
            total*= (n-i)
        for i in range(k):
            total/=(i+1)
        return total

    clf_dt = DecisionTreeClassifier(random_state=0)
    def score_comb_by_dt(dims):
        idxs = [x[0] for x in dims]
        rf_score = cross_val_score(clf_dt, X[:,idxs], dataY, cv=5)
        return np.mean(rf_score)

    def score_comb_by_cover(dims):
        sizes = np.array([len(x[1]) for x in dims])
        return sum(sizes)/max_sample_separation
    TOTAL_COMB = count_combinations(len(dims_cover),N_COMB)

    def score_by_coocorrency(dims):
        idxs = [x[0] for x in dims]
        ocorrencies = (X[:,idxs] >= CO_OCORRENCIE_LIMIAR)
        # ocorrencies = np.sum(ocorrencies, axis=1) / X.shape[1]
        count_ocorrencies = np.sum(ocorrencies, axis=1) >= len(idxs)
        return np.sum(count_ocorrencies)/X.shape[0]


    def score_by_regression_coocorrency(dims):
        idxs = [x[0] for x in dims]
        t_score = []
        for idx in idxs:
            mask = list(set(idxs)-set([idx]))
            reg = LinearRegression().fit(X[:,mask], X[:,idx])
            pred = reg.predict(X[:,mask])
            s = r2_score(X[:,idx], pred)

            if reg.coef_[0] < 0:
                s = -s
            t_score.append(s)

        # print("\n",min(np.sum(X[:,idxs], axis=0)),"\n")
        # return np.mean(t_score)
        return min(t_score)

    def score_by_scipy_stats(dims, stat_function):
        idxs = [x[0] for x in dims]
        if len(idxs) != 2:
            print("Error: Total of dimension combinations must be two for Pearson correlation")
            exit()
        corr, s = stat_function(X[:,idxs[0]], X[:,idxs[1]])
        s = s

        # if corr < 0:
        #     s = -s
        return corr
        # return (1-s)*corr

    def score_by_pearson(dims):
        return score_by_scipy_stats(dims,stats.pearsonr)
    def score_by_spearman(dims):
        return score_by_scipy_stats(dims,stats.spearmanr)
    def score_by_kendall(dims):
        return score_by_scipy_stats(dims,stats.kendalltau)

    scores = []
    edge_count = 0

    # Score functions must accept an set of dimensions (indexes) and X (Data)
    # and return the dimensions correlation probabilitiy

    if score_type == "dt":
        score_function = score_comb_by_dt
    elif score_type == "cover":
        score_function = score_comb_by_cover
    elif score_type == "coc":
        score_function = score_by_coocorrency
    elif score_type == "reg":
        score_function = score_by_regression_coocorrency
    elif score_type == "pearson":
        score_function = score_by_pearson
    elif score_type == "spearman":
        score_function = score_by_spearman
    elif score_type == "kendall":
        score_function = score_by_kendall
    

    def compute_score_thread(comb):
        dims = dims_cover[list(comb)]
        return score_function(dims)
        
    with Pool(cores) as p:
        scores = list(tqdm(p.imap(compute_score_thread, list(total_combinations)),
                           total=TOTAL_COMB,
                           desc="Computing scores... "))
    

    total_combinations = combinations(range(len(dims_cover)),N_COMB)
    iterator = zip(scores,total_combinations)
    iterator = tqdm(iterator, total=TOTAL_COMB, desc="Computing correlations...")

    filtered_scores = []
    for score_comb, comb in iterator:
        dims = dims_cover[list(comb)]
        # scores.append(score_comb)
        if(abs(score_comb) >= LIMIAR):
            filtered_scores.append(score_comb)
            edge_count+=1
            for i,j in combinations(range(len(dims)),2):
                GRAPH_ORGS[dims[i][0]].append((dims[j][0],score_comb))
                GRAPH_ORGS[dims[j][0]].append((dims[i][0],score_comb))
                # GRAPH_ORGS[dims[i][0]] = GRAPH_ORGS[dims[i][0]].union({dims[j][0]})
                # GRAPH_ORGS[dims[j][0]] = GRAPH_ORGS[dims[j][0]].union({dims[i][0]})
                
            
    print("Total edges added:",edge_count)
    # print("TOP 5 scores:",np.array(sorted(scores))[-5:])

    # number_classes

    def intra_extra_class_metric_class(dim_values, labels, c):
        # mean_class = np.mean(dim_values[labels[labels==c]])
        min_dis = max(dim_values) - min(dim_values)
        
        for i,x in enumerate(dim_values):
            if(labels[i] != c):
                continue
            for j,y in enumerate(dim_values):
                if(labels[j] == c):
                    continue
                min_dis = min(min_dis, abs(x-y))
        # if(min_dis != 0):
        #     print(dim_values[c==labels],dim_values[c!=labels], min_dis)
        return min_dis

    # Mean distance between minimum distances of each instance of a certain class
    # and any other instance from a different class
    def class_mean_distance(dim_values, labels, c):
        distances = []
        # for each instance of tested class
        for i,x in enumerate(dim_values):
            if(labels[i] != c):
                continue
            
            min_dis = max(dim_values) - min(dim_values)
            # for each instance that have a different class of the tested class
            for j,y in enumerate(dim_values):
                if(labels[j] == c):
                    continue
                min_dis = min(min_dis, abs(x-y))
            distances.append(min_dis)

        return np.mean(distances)

    LIMIAR_SET_GROUP = 0.0
    def get_group_class_from_dim(dim_values, labels):
        # scores = [intra_extra_class_metric_class(dim_values,labels,i) for i in range(number_classes)]
        scores = [class_mean_distance(dim_values,labels,i) for i in range(number_classes)]
        if(max(scores) == 0):
            return -1
        scores = np.array([x/max(scores) for x in scores])
        
        if(max(scores) >= LIMIAR_SET_GROUP):
            # print(list(zip(dim_values,labels)),scores)
            # print(scores,scores.index(max(scores)))
            return np.where(scores == scores.max())[0][-1]
        return -1

    def class_id_2_color(class_id):
        if(class_id == -1):
            return "rgba(0,0,0,0.1)"
        
        name_class = db.label_encoder.inverse_transform([class_id])[0]
        if name_class == "UASB1":
            return "rgba(31, 119, 180, 0.8)" # "blue"
        elif name_class == "UASB2":
            return "rgba(214, 39, 40, 0.8)" # "red"
        elif name_class == "CSRT1":
            return "rgba(255, 127, 14)" # "orange"
        elif name_class == "CSRT2":
            return "rgba(44, 160, 44)" # "green"
        elif name_class == "CSRT3":
            return "rgba(148, 103, 189)" # "purple"
        


    labels = {i:db.feature_names[i] for i in GRAPH_ORGS.keys()}

    graphData = {"nodes":[], "edges":[]}

    max_score = np.max(filtered_scores)
    min_score = np.min(filtered_scores)
    for x in GRAPH_ORGS:
        class_dim = get_group_class_from_dim(X[:,x], dataY)
        graphData["nodes"].append({
                "id": x,
                "label": labels[x],
                "color": {"border": class_id_2_color(class_dim)} 
        })
        for y in GRAPH_ORGS[x]:
            graphData["edges"].append({
                "from": x,
                "to": y[0],
                "value": ((min_score < 0)*(-min_score) + y[1])/((min_score < 0)*(-min_score) + max_score),
                "score": y[1],
            })

            if y[1] < 0:
                graphData["edges"][-1]["dashes"] = [1,1]

    with open('json/graphData_'+JSON_NAME+'.json', 'w') as fp:
        json.dump(graphData, fp)