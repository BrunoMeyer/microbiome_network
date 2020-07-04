"""
MIT License

Copyright (c) 2019 Bruno Henrique Meyer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from sklearn.linear_model import LassoCV
from sklearn.datasets import make_regression

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
from matplotlib import colors as mcolors
tableau_colors = list(mcolors.TABLEAU_COLORS.values())


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
# from math import comb

from external_dataset import load_biogas, load_age_database
from external_dataset import load_feature_importances


"""
Synonimous used in this code:
feature dimension = dimension = otu = taxon : Taxonomic group. Can represent any taxonomic rank

sample = class : The sample that is associated with each duplication/repetiton

duplication = instance : Each instance that is relationed with a sample
"""


def count_combinations(n,k):
    total = 1
    for i in range(k):
        total*= (n-i)
    for i in range(k):
        total/=(i+1)
    return total

def score_comb_by_dt(dims, X):
    idxs = [x[0] for x in dims]
    rf_score = cross_val_score(clf_dt, X[:,idxs], dataY, cv=5)
    return np.mean(rf_score)

def score_comb_by_cover(dims, X):
    sizes = np.array([len(x[1]) for x in dims])
    return sum(sizes)/max_sample_separation


def score_by_coocorrency(dims, X):
    idxs = [x[0] for x in dims]
    ocorrencies = (X[:,idxs] >= CO_OCORRENCIE_LIMIAR)
    count_ocorrencies = np.sum(ocorrencies, axis=1) >= len(idxs)
    return np.sum(count_ocorrencies)/X.shape[0]


def score_by_regression_coocorrency(dims, X):
    idxs = [x[0] for x in dims]
    t_score = []
    p_values = []
    for idx in idxs:
        mask = list(set(idxs)-set([idx]))
        # Regression with sklearn
        # reg = LinearRegression().fit(X[:,mask], X[:,idx])
        # pred = reg.predict(X[:,mask])
        # s = r2_score(X[:,idx], pred)
        # if reg.coef_[0] < 0:
        #     s = -s
        # t_score.append(s)

        slope, intercept, r_value, p_value, std_err = stats.linregress(X[:,mask[0]],X[:,idx])
        t_score.append(r_value)
        p_values.append(p_value)

    # Return the minimum value between correlation of (x,y) and (y,x)
    # Usually, this must be the same values
    return min(t_score), p_values[t_score.index(min(t_score))]

# There is a similar way to get correlation coefficients of kendall, spearman and pearson
# with scipy library. The next functions generalizes it 
def score_by_scipy_stats(dims, X, stat_function):
    '''
    Returns the correlation strenght and p-value
    '''
    idxs = [x[0] for x in dims]
    if len(idxs) != 2:
        print("Error: Total of dimension combinations must be two for {} correlation".format(
                str(stat_function)))
        exit(1)
    corr, s = stat_function(X[:,idxs[0]], X[:,idxs[1]])
    return corr, s

def score_by_pearson(dims, X):
    return score_by_scipy_stats(dims,X,stats.pearsonr)
def score_by_spearman(dims, X):
    return score_by_scipy_stats(dims,X,stats.spearmanr)
def score_by_kendall(dims, X):
    return score_by_scipy_stats(dims,X,stats.kendalltau)

# Used for development
# Plot points represented as a machine learning classification problem
def plot_samples(dataX,dataY,labels):
    plt.subplot()
    plt.title("")
    plt.scatter(dataX, dataY, marker='o', c=labels,
                s=50, edgecolor='k')
    plt.show()


# Intersection is metric used to sort the feature dimensions (taxons) by
# it importance to discriminate the samples
# Actually this is not used 
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
        heuristic_sum += relative_dis
    return heuristic_sum


# Compute the instances that is not discriminated for a specific feature dimension
# and sample
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
    return set(instances_covered)

class CorrelationGraphOptions(object):
    def __init__(self,
                 dims_cover,
                 graph_features,
                 score_function,
                 score_type,
                 include_p_val_scores,
                 LIMIAR,
                 p_val_limiar):
        self.dims_cover = dims_cover
        self.graph_features = graph_features
        self.score_function = score_function
        self.score_type = score_type
        self.include_p_val_scores = include_p_val_scores
        self.LIMIAR = LIMIAR
        self.p_val_limiar = p_val_limiar

        # Since the size of dataX, the transference between threads (cores)
        # can be a bottleneck in the computational time
        # self.dataX = dataX

__global_dataX__ = None

def compute_score_thread(args):
    # global __global_dataX__

    graph_opt, dim_id = args

    GRAPH_ORGS = defaultdict(list)
    for i in range(graph_opt.graph_features):
        if dim_id <= i:
            continue
        dims = graph_opt.dims_cover[[dim_id, i]]
        score_comb = graph_opt.score_function(dims, __global_dataX__)
        
        # By default ignore the p-value
        p_val = 0.0
        # If necessary, use the p-value limiar
        if graph_opt.score_type in graph_opt.include_p_val_scores:
            p_val = score_comb[1]
            score_comb = score_comb[0]
        
        # If p-value is accepted and the correlation is strong enough (positive or negative)
        if(abs(score_comb) >= graph_opt.LIMIAR and p_val <= graph_opt.p_val_limiar):
            # Add the edge to graph linking the features
            for i,j in combinations(range(len(dims)),2):
                # Add two bidirectional edge
                GRAPH_ORGS[dims[i][0]].append((dims[j][0],score_comb))
                GRAPH_ORGS[dims[j][0]].append((dims[i][0],score_comb))
    return GRAPH_ORGS

# Compute the minimum distance from a instance of a specifc class c to each
# other instance (rather than the class c),and returns the minimum distance
# found
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

# Compute the most discriminative class (sample) of specific feature dimension
def get_group_class_from_dim(dim_values,
                             labels,
                             number_classes,
                             LIMIAR_SET_GROUP = 0.0):
    # scores = [intra_extra_class_metric_class(dim_values,labels,i) for i in range(number_classes)]
    scores = [class_mean_distance(dim_values,labels,i) for i in range(number_classes)]
    if(max(scores) == 0):
        return -1
    scores = np.array([x/max(scores) for x in scores])
    
    if(max(scores) >= LIMIAR_SET_GROUP):
        return np.where(scores == scores.max())[0][-1]
    return -1

# TODO : Implement generic and deterministic color attribuction
# Assign a color for each sample
def class_id_2_color(class_id, class_id_2_name):
    if(class_id == -1):
        return "rgba(0,0,0,0.1)"
    
    name_class = class_id_2_name[class_id]
    color_val = tableau_colors[class_id]
    return color_val
    
    '''
    if name_class == "UASB1":
        return "rgba(31, 119, 180, 0.8)" # "blue"
    elif name_class == "UASB2":
        return "rgba(214, 39, 40, 0.8)" # "red"
    elif name_class == "CSTR1":
        return "rgba(255, 127, 14)" # "orange"
    elif name_class == "CSTR2":
        return "rgba(44, 160, 44)" # "green"
    elif name_class == "CSTR3":
        return "rgba(148, 103, 189)" # "purple"
    '''

def create_json_correlation(score_type,
                            abundance_limiar,
                            LIMIAR_ALPHA,
                            LIMIAR_BETA,
                            p_val_limiar,
                            only_important_features,
                            db,
                            X,
                            dataY,
                            feature_names,
                            features_importances,
                            cores=None):
    
    if cores is None:
        cores = multiprocessing.cpu_count()

    if only_important_features:
        # Filter tuples by the feature importance value (second position in tuple)
        features_importances = sorted(features_importances,
                                      key=lambda ft: ft[1], reverse=True)
        features_importances = np.array(features_importances)
        if only_important_features >= X.shape[1]:
            raise Exception("only_important_features({}) must be less than the number of features ({})".format(
                             only_important_features, X.shape[1]))
        
        # Get the id of selected features and filters the database
        important_features = features_importances[:only_important_features,0]
        important_features = np.array(important_features,
                                           dtype=np.int)
        
        # X = X[:, mask]
        # feature_names = feature_names[mask]


    # Normalize by column
    # for i in range(X.shape[1]):
    #     X[:,i] = X[:,i]/max(X[:,i])

    all_classes = set(dataY)
    number_classes = len(all_classes)
    total = set()

    edges = []

    # Creates a structure with the data of each dimension (otu) and relative abundances
    # observed in each sample
    # Also, compute a preliminar value (dim_intersect) that represent a
    # preliminar discriminative power of each dimension
    # Note: This is not necessary, but it was preserved for future works 
    dims_cover = []
    for i in range(X.shape[1]):
        dim_cover = dim_instance_separability(
            X[:,i],
            dataY
        )
        dim_intersect = multiclass_intersect_size(
            X[:,i],dataY, number_classes
        )

        if not i in important_features:
            continue
        dims_cover.append(
            [i,dim_cover,dim_intersect]
        )
        total = total.union(dim_instance_separability(X[:,i],dataY))

    max_sample_separation = len(set(total))

    # Sort by its preliminar importance
    dims_cover = sorted(dims_cover, key=lambda dim: dim[2], reverse=True)


    # This part of program consider different types of score type
    # The score type represents a metric used to estimate the correlation coefficient
    # for two differents feature dimensions. The most basic is spearman or pearson correlation


    # Only used in "coc" score type
    CO_OCORRENCIE_LIMIAR = LIMIAR_BETA
    if CO_OCORRENCIE_LIMIAR is None:
        CO_OCORRENCIE_LIMIAR = 0.8
    
    # N_COMB represents the number of features used for compute combinations
    # When N_COMB is 2, the features will be compared pairwise with score_type
    # The actual version only consider methods of score_type for comparison of
    # two features dimensions. Future works can explore this parameter
    # Note that the increase of N_COMB represent a exponentially increase of
    # possible combinations and also the time required for execution
    N_COMB = 2
    
    
    INT_ORGS = set()
    PLOT = False
    dims_cover = np.array(dims_cover)
    graph_features = len(dims_cover)

    # total_combinations = combinations(range(len(dims_cover)),N_COMB)
    if score_type == "dt":
        pass
    elif score_type == "cover":
        pass
    elif score_type == "coc":
        if LIMIAR_ALPHA is None:
            LIMIAR = 2.0/graph_features
        else:
            LIMIAR = float(LIMIAR_ALPHA)/X.shape[0]
    elif score_type=="reg" or score_type=="pearson" or score_type=="spearman" or score_type=="kendall":
        if LIMIAR_ALPHA is None:
            LIMIAR = 0.95
        else:
            LIMIAR = float(LIMIAR_ALPHA)
            
    print("Testing to",db.name," with",N_COMB,"combinations and limiar equal to",LIMIAR, "with",score_type," score type")
    print("Max cover "+str(max_sample_separation)+" from "+str(len(X)))


    # TOTAL_COMB = count_combinations(len(dims_cover),N_COMB)
    TOTAL_COMB = int(((graph_features**2) - graph_features)/2)


    # Score type by decision Tree Machine learning prediction
    # Two features have a high score if they have a high capacity of represenation
    # for discriminate samples
    clf_dt = DecisionTreeClassifier(random_state=0)
    

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
    
    # This scores types will consider the p-value limiar argument
    include_p_val_scores = ["pearson", "spearman", "kendall", "reg"]
    
    
    # Parallel processing of score types
    '''
    GRAPH_ORGS = defaultdict(list)

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
        
        # By default ignore the p-value
        p_val = 0.0
        # If necessary, use the p-value limiar
        if score_type in include_p_val_scores:
            p_val = score_comb[1]
            score_comb = score_comb[0]
        
        # If p-value is accepted and the correlation is strong enough (positive or negative)
        if(abs(score_comb) >= LIMIAR and p_val <= p_val_limiar):
            # Add the edge to graph linking the features
            filtered_scores.append(score_comb)
            edge_count+=1
            for i,j in combinations(range(len(dims)),2):
                # Add two bidirectional edge
                GRAPH_ORGS[dims[i][0]].append((dims[j][0],score_comb))
                GRAPH_ORGS[dims[j][0]].append((dims[i][0],score_comb))
    
    # Note: The graph generated will only consider nodes that have at least one edge

    print("Total edges added:",edge_count)
    # print("TOP 5 scores:",np.array(sorted(scores))[-5:])
    '''
    
    

    filtered_scores = []
    
    global __global_dataX__
    __global_dataX__ = X
    
    graph_opt = CorrelationGraphOptions(dims_cover,
                                        graph_features,
                                        score_function,
                                        score_type,
                                        include_p_val_scores,
                                        LIMIAR,
                                        p_val_limiar)

    GRAPH_ORGS = defaultdict(list)
    with Pool(cores) as p:
        arguments = [(graph_opt, dim_id)
                     for dim_id in list(range(graph_features))]
        threads_result = list(tqdm(p.imap(compute_score_thread, arguments),
                              total=graph_features,
                              desc="Computing scores... "))
        for go in threads_result:
            for key in go:
                GRAPH_ORGS[key]+=go[key]
                edge_count+=len(go[key])
                filtered_scores+=[x[1] for x in go[key]]
    # Note: The graph generated will only consider nodes that have at least one edge

    print("Total edges added:",edge_count)
    # print("TOP 5 scores:",np.array(sorted(scores))[-5:])


    


    # Create the graph/network as a json file and save it
    # The file can be interpreted by graph.html file
    labels = {i:feature_names[i] for i in GRAPH_ORGS.keys()}

    graphData = {"nodes":[], "edges":[], "border_legend":[]}

    class_id_2_name = {class_id: db.label_encoder.inverse_transform([class_id])[0]
                       for class_id in set(dataY)}
    
    
    for class_id in list(set(dataY)):
        class_name = class_id_2_name[class_id]
        color_val = tableau_colors[class_id]
        graphData["border_legend"].append([class_name,color_val])
    
    max_score = np.max(filtered_scores)
    min_score = np.min(filtered_scores)
    for x in GRAPH_ORGS:
        class_dim = get_group_class_from_dim(X[:,x], dataY, number_classes)
        graphData["nodes"].append({
                "id": x,
                "label": labels[x],
                "color": {"border": class_id_2_color(class_dim, class_id_2_name)} 
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

    
    return graphData

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
    parser.add_argument('--p_val_limiar', action = 'store', dest = 'p_val_limiar',
                                    default = 0.05, required = False, type=float,
                                    help = 'The p-value limiar used to filter the network correlation edges. Only works with "pearson", "kendall" and "spearman.')
    parser.add_argument('--workers', action = 'store', dest = 'workers',
                                    default = None, required = False, type=int,
                                    help = 'Number of workers used to parallelize processing.')
    parser.add_argument('--oif', '--only_important_features',
                        action = 'store', dest = 'only_important_features',
                        default=0, required = False, type=int,
                        help='Use only the N most important features.\n'+
                             'If not specified, the program will use all features.\n'+
                             'Must be less than the number of features.\n'
                             'Also, the file json/feature_importances must be '+
                             'previously created with otu_importances.py.\n'+
                             'If specified, the selector by id in the graph page '+
                             'cannot be trusted (possible broken).')
    
    arguments = parser.parse_args()
    cores = arguments.workers
    
    score_type = arguments.score_type
    taxa_level = arguments.taxa_level
    label_type = arguments.label_type
    abundance_limiar = arguments.abundance_limiar
    LIMIAR_ALPHA = arguments.LIMIAR_ALPHA
    LIMIAR_BETA = arguments.LIMIAR_BETA
    p_val_limiar = arguments.p_val_limiar
    only_important_features = arguments.only_important_features

    filter_by_taxa_level = None
    if not (arguments.filter_by_taxa_name is None) and not (arguments.filter_by_taxa_level is None):
        filter_by_taxa_level = (arguments.filter_by_taxa_level, arguments.filter_by_taxa_name)
    
    dtype = arguments.dtype
    JSON_NAME = arguments.json_name

    # Load the database considering the parameters specified by arguments of this program
    if dtype == "all_relative":
        db = load_biogas(data_type=dtype, label_type=label_type, label_value='name',
                        relative_taxa_level=taxa_level, abundance_limiar=abundance_limiar,
                        filter_by_taxa_level=filter_by_taxa_level)
    if dtype == "age":
        db = load_age_database()
    # python3 otu_correlation.py --json_name age --limiar_alpha 0.8 --score_type spearman
    X, dataY = (db.data, db.target)
    feature_names = db.feature_names

    features_importances = load_feature_importances(db.name)

    graphData = create_json_correlation(score_type,
                                        abundance_limiar,
                                        LIMIAR_ALPHA,
                                        LIMIAR_BETA,
                                        p_val_limiar,
                                        only_important_features,
                                        db,
                                        X,
                                        dataY,
                                        feature_names,
                                        features_importances)

    with open('json/graphData_'+JSON_NAME+'.json', 'w') as fp:
        json.dump(graphData, fp)