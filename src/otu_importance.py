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

from external_dataset import load_biogas, load_age_database

from metrics import ClassifierMetrics, ClassifierMetricsSet
from sklearn import datasets


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC


from sklearn.model_selection import StratifiedKFold

import time

from collections import defaultdict


from sklearn.kernel_approximation import RBFSampler
from sklearn.svm import SVR
from sklearn.feature_selection import RFE


from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AffinityPropagation

import sys
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

from sklearn import tree


import json
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, chi2



from tqdm import tqdm

import multiprocessing
import argparse
cores = multiprocessing.cpu_count()


"""
Synonimous used in this code:
feature dimension = dimension = otu = taxon : Taxonomic group. Can represent any taxonomic rank

sample = class : The sample that is associated with each duplication/repetiton

duplication = instance : Each instance that is relationed with a sample
"""






def create_json_importance(databases,
                           TYPE_SELECTION="RF",
                           NORMALIZE_DATA_BEFORE=False,
                           RANDOM_STATE = 0):
    '''
    # Algorithm used for compute feature importance
        # TYPE_SELECTION = "SVM-RFE"
        TYPE_SELECTION = "RF"
    '''
    feature_importances_json = {}
    # feature_importances_json["_taxa_levels_hierarchy"] = biogas_database_relative_grouped_0.taxa_levels_hierarchy
    feature_importances_json["_taxa_levels_hierarchy"] = databases[0].taxa_levels_hierarchy

    iterator = tqdm(databases, total=len(databases),
                    desc="Computing feature importances for each database")

    for database in iterator:
        db_name = database.name
        dataX, dataY, dataY_value = (database.data,
                                     database.target,
                                     database.target_values)
        dataX = dataX[:,:]
        if(NORMALIZE_DATA_BEFORE):
            # dataX = preprocessing.normalize(dataX, norm='l1', axis=0)
            # for i in range(dataX.shape[0]):
            #     dataX[i,:] = dataX[i,:]/sum(dataX[i,:])
            for i in range(dataX.shape[1]):
                dataX[:,i] = dataX[:,i]/max(dataX[:,i])
            dataX[np.isnan(dataX)] = 0.0
        
        # Compute the feature importance for each otu
        if(TYPE_SELECTION == "SVM-RFE"):
            estimator = SVR(kernel="linear")
            selector_svm_rfe = RFE(estimator, 1, step=1)
            selector_svm_rfe.fit(dataX,dataY)
            fi = selector_svm_rfe.ranking_
            fi = [len(fi) - float(f) for f in fi]            

        
        if(TYPE_SELECTION == "RF"):
            clf_etc = ExtraTreesClassifier(n_estimators=500, bootstrap=False,
                                            oob_score=False, n_jobs=cores,
                                            random_state=RANDOM_STATE)
            clf_etc = clf_etc.fit(dataX, dataY)

            fi = clf_etc.feature_importances_
        
        # TODO: Create an option to normalize the feature importances
        # max_value_fi = max([abs(x) for x in fi])
        
        
        feature_importances_json[db_name] = defaultdict(list)
        
        # Create a relation between each otu and a index
        for i, score in enumerate(fi):
            feature_importances_json[db_name]["scores_rf"].append(
                [i, score]
            )
        
        # The databases with "grouped" tag in it name represent the datasets
        # with multi-class problem
        # The standard dataset contains attributes relationed with each class
        # In this part, the importance of each feature is computed considering
        # it impact when a regressor is used to predict this attributes
        # The attributes can represent pH, Ammonia and others
        if("grouped" in db_name):
            for label_value_type in database.all_target_values:
                dataY_value = database.all_target_values[label_value_type]

                reg = LinearRegression().fit(dataX, dataY_value)
                coefs_reg = reg.coef_
                max_value_reg = sum(abs(coefs_reg))
                for i, score in enumerate(coefs_reg):
                    feature_importances_json[db_name]["scores_reg_"+label_value_type].append(
                        [i, abs(score)/max_value_reg]
                    )
                    feature_importances_json[db_name]["target_values_"+label_value_type] = dataY_value

        # Save the data in json file. The graph.html, taxon_importance.html
        # and scores.html use the json created
        # TODO: There are many redudancies that can be optimized
        feature_importances_json[db_name]["dataX"] = dataX.tolist()
        decoded_labels = database.label_encoder.inverse_transform(dataY)
        feature_importances_json[db_name]["dataY"] = decoded_labels.tolist()
        feature_importances_json[db_name]["feature_names"] = database.feature_names.tolist()
        feature_importances_json[db_name]["feature_ids"] = list(range(len(database.feature_names.tolist())))
        feature_importances_json[db_name]["label_description"] = database.label_description
        feature_importances_json[db_name]["target_values"] = database.target_values
        
        # TODO: The next two functions are also defined in otu_correlation.py
        #       Its important to create a unique module

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
            number_classes = len(set(labels))
            scores = [class_mean_distance(dim_values,labels,i) for i in range(number_classes)]
            
            if(max(scores) == 0):
                return labels[np.argmax(dim_values)]
            scores = np.array([x/max(scores) for x in scores])
            
            if(max(scores) >= LIMIAR_SET_GROUP):
                return np.where(scores == scores.max())[0][-1]
            return -1

        # Get the real name of feature and compute the most discriminative sample for each otu
        dcdl = decoded_labels.tolist()
        discriminated_class = []
        for i in range(dataX.shape[1]):
            d = get_group_class_from_dim(dataX[:,i],dataY)
            d = database.label_encoder.inverse_transform([d])[0]

            discriminated_class.append(d)

    
        feature_importances_json[db_name]["discriminated_class"] = discriminated_class

    return feature_importances_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Transform the result from SparCC, Kendall, Spearman and Pearson (.out files) as graph into json files.')
    parser.add_argument('--normalize', action = 'store_true', dest = 'normalize',
                                required = False,
                                help = 'Normalize data before use Feature Selection model')
    arguments = parser.parse_args()

    NORMALIZE_DATA_BEFORE = arguments.normalize


    # Utilize the metane production as label (usefull for regression analysis)
    # label_value = "LN.biogás.kg SV-1"

    # Utilize the name of sample group as label
    label_value = "name"


    # biogas_database_bacteria_grouped = load_biogas(data_type = "bacteria", label_type="grouped", label_value=label_value)
    # biogas_database_archea_grouped = load_biogas(data_type = "archea", label_type="grouped", label_value=label_value)
    # biogas_database_bacteria_grouped_fill = load_biogas(data_type = "bacteria", label_type="grouped", label_value=label_value, abundance_limiar=0.05)
    # biogas_database_archea_grouped_fill = load_biogas(data_type = "archea", label_type="grouped", label_value=label_value, abundance_limiar=0.05)
    # biogas_database_bioem_grouped = load_biogas(data_type = "bioem", label_type="grouped", label_value=label_value)
    # biogas_database_merged_grouped = load_biogas(data_type = "merged", label_type="grouped", label_value=label_value)
    
    # Load the dataset considering different aspects
    '''
    biogas_database_relative_grouped_0 = load_biogas(data_type = "all_relative", label_type="grouped", label_value=label_value, relative_taxa_level=0)
    biogas_database_relative_grouped_1 = load_biogas(data_type = "all_relative", label_type="grouped", label_value=label_value, relative_taxa_level=1)
    biogas_database_relative_grouped_2 = load_biogas(data_type = "all_relative", label_type="grouped", label_value=label_value, relative_taxa_level=2)
    biogas_database_relative_grouped_3 = load_biogas(data_type = "all_relative", label_type="grouped", label_value=label_value, relative_taxa_level=3)
    biogas_database_relative_grouped_4 = load_biogas(data_type = "all_relative", label_type="grouped", label_value=label_value, relative_taxa_level=4)
    biogas_database_relative_grouped_5 = load_biogas(data_type = "all_relative", label_type="grouped", label_value=label_value, relative_taxa_level=5)
    biogas_database_relative_grouped_5_arch = load_biogas(data_type = "all_relative", label_type="grouped", label_value=label_value, relative_taxa_level=5, filter_by_taxa_level=(0,"Archaea"))
    biogas_database_relative_grouped_5_bact = load_biogas(data_type = "all_relative", label_type="grouped", label_value=label_value, relative_taxa_level=5, filter_by_taxa_level=(0,"Bacteria"))
    biogas_database_relative_grouped_0_fill = load_biogas(data_type = "all_relative", label_type="grouped", label_value=label_value, relative_taxa_level=0,abundance_limiar=0.05)
    biogas_database_relative_grouped_1_fill = load_biogas(data_type = "all_relative", label_type="grouped", label_value=label_value, relative_taxa_level=1,abundance_limiar=0.05)
    biogas_database_relative_grouped_2_fill = load_biogas(data_type = "all_relative", label_type="grouped", label_value=label_value, relative_taxa_level=2,abundance_limiar=0.05)
    biogas_database_relative_grouped_3_fill = load_biogas(data_type = "all_relative", label_type="grouped", label_value=label_value, relative_taxa_level=3,abundance_limiar=0.05)
    biogas_database_relative_grouped_4_fill = load_biogas(data_type = "all_relative", label_type="grouped", label_value=label_value, relative_taxa_level=4,abundance_limiar=0.05)
    biogas_database_relative_grouped_5_fill = load_biogas(data_type = "all_relative", label_type="grouped", label_value=label_value, relative_taxa_level=5,abundance_limiar=0.05)
    biogas_database_relative_grouped_5_arch_fill = load_biogas(data_type = "all_relative", label_type="grouped", label_value=label_value, relative_taxa_level=5,abundance_limiar=0.05, filter_by_taxa_level=(0,"Archaea"))
    biogas_database_relative_grouped_5_bact_fill = load_biogas(data_type = "all_relative", label_type="grouped", label_value=label_value, relative_taxa_level=5,abundance_limiar=0.05, filter_by_taxa_level=(0,"Bacteria"))

    biogas_database_bacteria_binary = load_biogas(data_type = "bacteria", label_type="binary")
    biogas_database_archea_binary = load_biogas(data_type = "archea", label_type="binary")
    biogas_database_bacteria_binary_fill = load_biogas(data_type = "bacteria", label_type="binary", abundance_limiar=0.05)
    biogas_database_archea_binary_fill = load_biogas(data_type = "archea", label_type="binary", abundance_limiar=0.05)
    # biogas_database_bioem_binary = load_biogas(data_type = "bioem", label_type="binary")
    # biogas_database_merged_binary = load_biogas(data_type = "merged", label_type="binary")
    biogas_database_relative_binary_0 = load_biogas(data_type = "all_relative", label_type="binary", relative_taxa_level=0)
    biogas_database_relative_binary_1 = load_biogas(data_type = "all_relative", label_type="binary", relative_taxa_level=1)
    biogas_database_relative_binary_2 = load_biogas(data_type = "all_relative", label_type="binary", relative_taxa_level=2)
    biogas_database_relative_binary_3 = load_biogas(data_type = "all_relative", label_type="binary", relative_taxa_level=3)
    biogas_database_relative_binary_4 = load_biogas(data_type = "all_relative", label_type="binary", relative_taxa_level=4)
    biogas_database_relative_binary_5 = load_biogas(data_type = "all_relative", label_type="binary", relative_taxa_level=5)
    biogas_database_relative_binary_5_arch = load_biogas(data_type = "all_relative", label_type="binary", relative_taxa_level=5, filter_by_taxa_level=(0,"Archaea"))
    biogas_database_relative_binary_5_bact = load_biogas(data_type = "all_relative", label_type="binary", relative_taxa_level=5, filter_by_taxa_level=(0,"Bacteria"))
    biogas_database_relative_binary_0_fill = load_biogas(data_type = "all_relative", label_type="binary", relative_taxa_level=0, abundance_limiar=0.05)
    biogas_database_relative_binary_1_fill = load_biogas(data_type = "all_relative", label_type="binary", relative_taxa_level=1, abundance_limiar=0.05)
    biogas_database_relative_binary_2_fill = load_biogas(data_type = "all_relative", label_type="binary", relative_taxa_level=2, abundance_limiar=0.05)
    biogas_database_relative_binary_3_fill = load_biogas(data_type = "all_relative", label_type="binary", relative_taxa_level=3, abundance_limiar=0.05)
    biogas_database_relative_binary_4_fill = load_biogas(data_type = "all_relative", label_type="binary", relative_taxa_level=4, abundance_limiar=0.05)
    biogas_database_relative_binary_5_fill = load_biogas(data_type = "all_relative", label_type="binary", relative_taxa_level=5, abundance_limiar=0.05)
    biogas_database_relative_binary_5_arch_fill = load_biogas(data_type = "all_relative", label_type="binary", relative_taxa_level=4, abundance_limiar=0.05, filter_by_taxa_level=(0,"Archaea"))
    biogas_database_relative_binary_5_bact_fill = load_biogas(data_type = "all_relative", label_type="binary", relative_taxa_level=5, abundance_limiar=0.05, filter_by_taxa_level=(0,"Bacteria"))
    '''
    age_dataset = load_age_database()


    '''
    databases = [
        # ("Bacteria-grouped", biogas_database_bacteria_grouped),
        # ("Archea-grouped", biogas_database_archea_grouped),
        # ("Bioem-grouped", biogas_database_bioem_grouped),
        # ("Merged-grouped", biogas_database_merged_grouped),
        ("Relative_taxa0-grouped",biogas_database_relative_grouped_0),
        ("Relative_taxa1-grouped",biogas_database_relative_grouped_1),
        ("Relative_taxa2-grouped",biogas_database_relative_grouped_2),
        ("Relative_taxa3-grouped",biogas_database_relative_grouped_3),
        ("Relative_taxa4-grouped",biogas_database_relative_grouped_4),
        ("Relative_taxa5-grouped",biogas_database_relative_grouped_5),
        ("Relative_taxa5_arch-grouped",biogas_database_relative_grouped_5_arch),
        ("Relative_taxa5_bact-grouped",biogas_database_relative_grouped_5_bact),
        
        # ("Bacteria-binary", biogas_database_bacteria_binary),
        # ("Archea-binary", biogas_database_archea_binary),
        # ("Bioem-binary", biogas_database_bioem_binary),
        # ("Merged-binary", biogas_database_merged_binary),
        ("Relative_taxa0-binary", biogas_database_relative_binary_0),
        ("Relative_taxa1-binary", biogas_database_relative_binary_1),
        ("Relative_taxa2-binary", biogas_database_relative_binary_2),
        ("Relative_taxa3-binary", biogas_database_relative_binary_3),
        ("Relative_taxa4-binary", biogas_database_relative_binary_4),
        ("Relative_taxa5-binary", biogas_database_relative_binary_5),
        ("Relative_taxa5_arch-binary", biogas_database_relative_binary_5_arch),
        ("Relative_taxa5_bact-binary", biogas_database_relative_binary_5_bact),



        # ("Bacteria_fill-grouped", biogas_database_bacteria_grouped_fill),
        # ("Archea_fill-grouped", biogas_database_archea_grouped_fill),
        ("Relative_taxa0_fill-grouped",biogas_database_relative_grouped_0_fill),
        ("Relative_taxa1_fill-grouped",biogas_database_relative_grouped_1_fill),
        ("Relative_taxa2_fill-grouped",biogas_database_relative_grouped_2_fill),
        ("Relative_taxa3_fill-grouped",biogas_database_relative_grouped_3_fill),
        ("Relative_taxa4_fill-grouped",biogas_database_relative_grouped_4_fill),
        ("Relative_taxa5_fill-grouped",biogas_database_relative_grouped_5_fill),
        ("Relative_taxa5_arch_fill-grouped",biogas_database_relative_grouped_5_arch_fill),
        ("Relative_taxa5_bact_fill-grouped",biogas_database_relative_grouped_5_bact_fill),
    
        # ("Bacteria_fill-binary", biogas_database_bacteria_binary_fill),
        # ("Archea_fill-binary", biogas_database_archea_binary_fill),
        ("Relative_taxa0_fill-binary", biogas_database_relative_binary_0_fill),
        ("Relative_taxa1_fill-binary", biogas_database_relative_binary_1_fill),
        ("Relative_taxa2_fill-binary", biogas_database_relative_binary_2_fill),
        ("Relative_taxa3_fill-binary", biogas_database_relative_binary_3_fill),
        ("Relative_taxa4_fill-binary", biogas_database_relative_binary_4_fill),
        ("Relative_taxa5_fill-binary", biogas_database_relative_binary_5_fill),
        # ("Relative_taxa5_arch_fill-binary", biogas_database_relative_binary_5_arch_fill),
        ("Relative_taxa5_bact_fill-binary", biogas_database_relative_binary_5_bact_fill),
    ]
    '''

    databases = [
        # ("age_dataset", age_dataset)
        age_dataset
    ]

    # TYPE_SELECTION = "SVM-RFE"
    TYPE_SELECTION = "RF"

    RANDOM_STATE = 0

    json_features_importances = create_json_importance(databases,
                                                       TYPE_SELECTION,
                                                       NORMALIZE_DATA_BEFORE,
                                                       RANDOM_STATE)

    # Finally, save the json file
    with open('json/feature_importances.json', 'w') as fp:
        json.dump(json_features_importances, fp)