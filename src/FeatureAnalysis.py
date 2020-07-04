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


import argparse
from multiprocessing import Pool
import multiprocessing
import json

from external_dataset import load_biogas, load_age_database
from external_dataset import load_feature_importances

from otu_importance import create_json_importance
from otu_correlation import create_json_correlation

from create_js_from_json import create_js_from_json

"""
Synonimous used in this code:
feature dimension = dimension = otu = taxon : Taxonomic group. Can represent any taxonomic rank

sample = class : The sample that is associated with each duplication/repetiton

duplication = instance : Each instance that is relationed with a sample
"""

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
    parser.add_argument('--normalize_otu_importance', action = 'store_true',
                        dest = 'normalize_otu_importance',
                        required = False,
                        help = 'Normalize data before use Feature Selection model')

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
    p_val_limiar = arguments.p_val_limiar
    only_important_features = arguments.only_important_features

    OTU_IMPORTANCE_NORMALIZE_DATA_BEFORE = arguments.normalize_otu_importance

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
    
    databases = [
        # ("age_dataset", age_dataset)
        # age_dataset
        db
    ]

    # TYPE_SELECTION = "SVM-RFE"
    TYPE_SELECTION = "RF"

    RANDOM_STATE = 0


    json_features_importances = create_json_importance(
        databases,
        TYPE_SELECTION,
        OTU_IMPORTANCE_NORMALIZE_DATA_BEFORE,
        RANDOM_STATE)

    # Finally, save the json file
    with open('json/feature_importances.json', 'w') as fp:
        json.dump(json_features_importances, fp)
        
    
    
    # python3 otu_correlation.py --json_name age --limiar_alpha 0.8 --score_type spearman

    for db in databases:
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

        with open('json/graphData_{}.json'.format(db.name), 'w') as fp:
            json.dump(graphData, fp)

    create_js_from_json()