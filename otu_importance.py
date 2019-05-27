from external_dataset import load_biogas
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









RANDOM_STATE = 0

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


    # TYPE_SELECTION = "SVM-RFE"
    TYPE_SELECTION = "RF"

    feature_importances_json = {}
    feature_importances_json["_taxa_levels_hierarchy"] = biogas_database_relative_grouped_0.taxa_levels_hierarchy

    iterator = tqdm(databases, total=len(databases),
                    desc="Computing feature importances for each database")

    for db_name, database in iterator:
        dataX, dataY, dataY_value = (database.data, database.target, database.target_values)
        dataX = dataX[:,:]
        if(NORMALIZE_DATA_BEFORE):
            # dataX = preprocessing.normalize(dataX, norm='l1', axis=0)
            # for i in range(dataX.shape[0]):
            #     dataX[i,:] = dataX[i,:]/sum(dataX[i,:])
            for i in range(dataX.shape[1]):
                dataX[:,i] = dataX[:,i]/max(dataX[:,i])
            dataX[np.isnan(dataX)] = 0.0
        
        if(TYPE_SELECTION == "SVM-RFE"):
            # estimator = LinearSVC(C=0.01, penalty="l1", dual=False)
            # estimator.fit(dataX,dataY)
            # fi = estimator.coef_
            # if len(fi.shape) > 1:
            #     fi = np.sum(fi,axis=1)
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
        
        max_value_fi = max([abs(x) for x in fi])
        
        
        feature_importances_json[db_name] = defaultdict(list)
            
        for i, score in enumerate(fi):
            feature_importances_json[db_name]["scores_rf"].append(
                [i, score]
            )
            
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

        feature_importances_json[db_name]["dataX"] = dataX.tolist()
        decoded_labels = database.label_encoder.inverse_transform(dataY)
        feature_importances_json[db_name]["dataY"] = decoded_labels.tolist()
        feature_importances_json[db_name]["feature_names"] = database.feature_names.tolist()
        feature_importances_json[db_name]["feature_ids"] = list(range(len(database.feature_names.tolist())))
        feature_importances_json[db_name]["label_description"] = database.label_description
        feature_importances_json[db_name]["target_values"] = database.target_values
        
        with open('json/feature_importances.json', 'w') as fp:
            json.dump(feature_importances_json, fp)