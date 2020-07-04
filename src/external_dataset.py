import csv
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder

from collections import defaultdict

import json

class DataSet():
    data = []
    target = []
    target_levels = []
    target_values = []
    target_names = []
    feature_names = []
    label_encoder = None
    label_description = {}
    all_target_values = []
    taxa_levels_hierarchy = None
    name = None

    def __init__(self):
        pass

def read_csv(path, delimiter=','):
    df = pd.read_csv(path, sep=delimiter, header=0)
    return list(df), df.values


def load_label_description():
    # Value type in 
    # header = [
    #     'Amostras', 'repetições', 'açucares redutores ', 'Proteína Total',
    #     'Condutividade', 'Temperatura das amostras ', 'Amonia', 'Sólidos Totais ',
    #     'Sólidos Fixos', 'Sólidos Volatéis', 'pH', 'Sólidos Sedimentáveis',
    #     'LN.biogás.kg SV-1', 'LN Metano. Kg-1 sv'
    # ]

    data = read_csv("data_sample/Methane replicates.csv")
    header = list(data[0])
    table = list(data[1])

    label_description = defaultdict(lambda:'')
    
    for value_type in header[2:]:
        values = defaultdict(list)
        index_value_type = header.index(value_type)
        for line in table:
            if(label_description[line[0]] == ''):
                label_description[line[0]] = line[0]+'<br>'
            
            values[ line[0] ].append( line[index_value_type] )

        values = {x:np.mean(values[x]) for x in values}

        for group in values:
            label_description[group] += value_type+'(Average): '+str(values[group])+"<br>"
    
    return dict(label_description)

def load_label_values(value_type):
    # Value type in 
    #[
    # 'Amostras', 'repetições', 'açucares redutores ', 'Proteína Total',
    # 'Condutividade', 'Temperatura das amostras ', 'Amonia', 'Sólidos Totais ',
    # 'Sólidos Fixos', 'Sólidos Volatéis', 'pH', 'Sólidos Sedimentáveis',
    # 'LN.biogás.kg SV-1', 'LN Metano. Kg-1 sv'
    #]

    data = read_csv("data_sample/Methane replicates.csv")
    header = list(data[0])
    table = list(data[1])
    values = defaultdict(list)

    index_value_type = header.index(value_type)
    for line in table:
        values[ line[0] ].append( line[index_value_type] )

    values = {x:np.mean(values[x]) for x in values}
    return values

def load_all_label_values():
    # Value type in 
    #[
    # 'Amostras', 'repetições', 'açucares redutores ', 'Proteína Total',
    # 'Condutividade', 'Temperatura das amostras ', 'Amonia', 'Sólidos Totais ',
    # 'Sólidos Fixos', 'Sólidos Volatéis', 'pH', 'Sólidos Sedimentáveis',
    # 'LN.biogás.kg SV-1', 'LN Metano. Kg-1 sv'
    #]

    data = read_csv("data_sample/Methane replicates.csv")
    header = list(data[0])
    table = list(data[1])
    label_values = {}
    for value_type in header[2:]:
        values = defaultdict(list)
        index_value_type = header.index(value_type)
        for line in table:
            values[ line[0] ].append( line[index_value_type] )

        values = {x:np.mean(values[x]) for x in values}
        label_values[value_type] = values
    return label_values


def biogas_type_to_class():
    return {
        "EV1-2": "UASB1",
        "EV1-3": "UASB1",
        "EV1-4": "UASB1",
        "EV1-5": "UASB1",
        "EV1-1": "UASB1",
        "EV2-1": "CSTR1",
        "EV2-2": "CSTR1",
        "EV2-3": "CSTR1",
        "EV2-4": "CSTR1",
        "EV2-5": "CSTR1",
        "EV3-10": "UASB2",
        "EV11": "CSTR3",
        "EV3-12": "CSTR3",
        "EV3-13": "CSTR3",
        "EV3-14": "CSTR3",
        "EV3-15": "CSTR3",
        "EV3-1": "CSTR2",
        "EV3-2": "CSTR2",
        "EV3-3": "CSTR2",
        "EV3-4": "CSTR2",
        "EV3-5": "CSTR2",
        "EV3-6": "UASB2",
        "EV3-7": "UASB2",
        "EV3-8": "UASB2",
        "EV3-9": "UASB2"
    }

def biogas_type_to_class_binary():
    return {
        "EV1-2": "UASB",
        "EV1-3": "UASB",
        "EV1-4": "UASB",
        "EV1-5": "UASB",
        "EV1-1": "UASB",
        "EV2-1": "CSTR",
        "EV2-2": "CSTR",
        "EV2-3": "CSTR",
        "EV2-4": "CSTR",
        "EV2-5": "CSTR",
        "EV3-10": "UASB",
        "EV11": "CSTR",
        "EV3-12": "CSTR",
        "EV3-13": "CSTR",
        "EV3-14": "CSTR",
        "EV3-15": "CSTR",
        "EV3-1": "CSTR",
        "EV3-2": "CSTR",
        "EV3-3": "CSTR",
        "EV3-4": "CSTR",
        "EV3-5": "CSTR",
        "EV3-6": "UASB",
        "EV3-7": "UASB",
        "EV3-8": "UASB",
        "EV3-9": "UASB"
    }


def load_feature_importances(db_name):
    with open("json/feature_importances.json", "r") as f:
        feature_importances_json = json.load(f)
    return feature_importances_json[db_name]["scores_rf"]




def load_age_database():
    header, dataset_data = read_csv("data_sample/gene_labelled_data.csv")
    
    idx = np.argwhere(np.all(dataset_data[..., :] == 0, axis=0))
    dataset_data = np.delete(dataset_data, idx, axis=1)
    idx = idx.reshape((1,-1))[0]
    header = np.delete(np.array(header),idx)
    
    idx = np.argwhere(np.all(dataset_data[..., :] == 0.0, axis=0))
    dataset_data = np.delete(dataset_data, idx, axis=1)
    idx = idx.reshape((1,-1))[0]
    header = np.delete(np.array(header),idx)
    
    # dataset_data = dataset_data[:,:50]
    # header = header[:50]

    ret = DataSet()
    ret.data = dataset_data[:,3:]

    
    raw_target = dataset_data[:,1]
    interval_val = 10
    target = np.array(raw_target, dtype=np.object)
    for i in range(0,max(target), interval_val):
        new_class_name = "Age between {} and {}".format(i,i+interval_val)
        target[(raw_target>i) & (raw_target <= i+interval_val)] = new_class_name
    
    ret.label_encoder = LabelEncoder()
    target = ret.label_encoder.fit_transform(target)
    ret.target = target
    ret.name = "Age"
    ret.feature_names = np.array(header[3:])
    # target = []
    # target_levels = []
    # target_values = []
    # target_names = []
    # feature_names = []
    # label_encoder = None
    # label_description = {}
    # all_target_values = []
    # taxa_levels_hierarchy = None
    # name = None

    return ret

#Biogas
def load_biogas(data_type="all_relative", label_type="grouped", label_value='name',
                merge_features=True, relative_taxa_level=-1, abundance_limiar=1.0,
                filter_by_taxa_level=None):

    if(label_type == "grouped"):
        label_transform = biogas_type_to_class()
    elif(label_type=="binary"):
        label_transform = biogas_type_to_class_binary()
    
    dataset_bacteria = read_csv("data_sample/Genera_otus_bacteria.csv")
    dataset_archea = read_csv("data_sample/Genera_otus_archea.csv")
    dataset_bioem = read_csv("data_sample/biogas_bioem.csv")
    dataset_all_relative = read_csv("data_sample/relative_abundances.csv")
    # print(dataset_all_relative[1].shape)
    
    taxa_levels_hierarchy = {}
    for i in range(1,6):
        taxa_levels_hierarchy[i] = {}
        for line in dataset_all_relative[1]:
            x = line[i].replace("D_"+str(i)+"__","")
            # y = line[i-1].replace("D_"+str(i-1)+"__","")
            if not (x in taxa_levels_hierarchy[i]):
                taxa_levels_hierarchy[i][x] = [line[j].replace("D_"+str(j)+"__","")
                                               for j in range(i)]
    

    if not (filter_by_taxa_level is None):
        level_filter = filter_by_taxa_level[0]
        filter_name = filter_by_taxa_level[1]
        dataset_all_relative = (dataset_all_relative[0],
                                np.array([x for i,x in enumerate(dataset_all_relative[1])
                                          if filter_name in x[level_filter]]))
    # print(dataset_all_relative[1].shape)
    # dataset = np.matrix(dataset[1])

    transposed_bacteria = dataset_bacteria[1].transpose()
    transposed_allr = dataset_all_relative[1].transpose()
    transposed_archea = dataset_archea[1].transpose()
    transposed_bioem = dataset_bioem[1].transpose()
    # [print(x) for x in transposed[0]]
    # print(dataset.shape)

    feature_names_bacteria = transposed_bacteria[0]
    feature_names_archea = transposed_archea[0]+transposed_archea[1]
    feature_names_bioem =  transposed_bioem[0]

    taxa_levels = {}
    for i in range(6):
        filtered_taxa = transposed_allr[i]
        filtered_taxa = [x.replace("D_"+str(i)+"__","")
                              for x in filtered_taxa]
        taxa_levels[i] = np.array(filtered_taxa)

    if relative_taxa_level == -1:
        feature_names_allr = transposed_allr[0]+transposed_allr[1]+\
                            transposed_allr[2]+transposed_allr[3]+\
                            transposed_allr[4]+transposed_allr[5]
    else:
        feature_names_allr = transposed_allr[relative_taxa_level]
        feature_names_allr = [x.replace("D_"+str(relative_taxa_level)+"__","")
                              for x in feature_names_allr]
        

    # print(dataset_all_relative[0])
    # print(feature_names_allr)
    new_feature_names = []
    for i,x in enumerate(feature_names_bacteria):
        if merge_features:
            new_feature_names.append(x)
        else:
            new_feature_names.append(x+" em Genera_otus_archea.xlsx(Bacteria) (Linha "+str(i)+")")
    feature_names_bacteria = np.array(new_feature_names)

    new_feature_names = []
    for i,x in enumerate(feature_names_allr):
        if merge_features:
            new_feature_names.append(x)
        else:
            new_feature_names.append(x+" em relative_abundances.xlsx(All relative abundances) (Linha "+str(i)+")")
    feature_names_allr = np.array(new_feature_names)
    
    new_feature_names = []
    for i,x in enumerate(feature_names_archea):
        if merge_features:
            new_feature_names.append(x)
        else:
            new_feature_names.append(x+" em Genera_otus_archea.xlsx(Archea) (Linha "+str(i)+")")
    feature_names_archea = np.array(new_feature_names)

    
    new_feature_names = []
    for i,x in enumerate(feature_names_bioem):
        if merge_features:
            new_feature_names.append(x)
        else:
            new_feature_names.append(x+" em table_even4530.biom (Linha "+str(i)+")")
    feature_names_bioem = np.array(new_feature_names)


    data_bacteria = transposed_bacteria[1:]
    data_allr = transposed_allr[6:]
    data_archea = transposed_archea[2:]
    data_bioem = transposed_bioem[1:]
    


    #TODO: Verify if the target is same
    
    # aux = dataset_bacteria[0]
    # del aux[0]
    # print([label_transform[x] for x in aux])
    # aux = dataset_bioem[0]
    # del aux[0]
    # print([label_transform[x] for x in aux])

    # Remove [0,0] position
    if data_type == "all_relative":
        target = dataset_all_relative[0][6:]
    else:
        target = dataset_bacteria[0]
        del target[0]

    target = [label_transform[x] for x in target]

    

    ret = DataSet()

    ret.name = data_type+"_"+label_type+"_"+label_value
    ret.target_names = target
    ret.target_levels = taxa_levels
    ret.taxa_levels_hierarchy = taxa_levels_hierarchy
    
    if(label_value != 'name'):
        ret.target_values = list(target)
        group_values = load_label_values(label_value)
        for i,x in enumerate(ret.target_values):
            ret.target_values[i] = group_values[x]
    
    ret.all_target_values = load_all_label_values()

    for label_value in ret.all_target_values:
        aux = list(target)
        all_group_values = ret.all_target_values[label_value]
        for i,x in enumerate(aux):
            if(x in all_group_values):
                aux[i] = all_group_values[x]
        ret.all_target_values[label_value] = aux


    
    le = LabelEncoder()
    target = le.fit_transform(target)
    ret.target = target
    ret.label_encoder = le


    # data = np.delete(data,index_class,1)
    ret.data = np.concatenate((
        data_bacteria,
        data_archea,
        data_bioem,
    ), axis=1)

    ret.feature_names = np.concatenate((
        feature_names_bacteria,
        feature_names_archea,
        feature_names_bioem,
    ))

    ret.label_description = load_label_description()

    if(data_type == "bacteria"):
        ret.data = data_bacteria
        ret.feature_names = feature_names_bacteria
    
    if(data_type == "bioem"):
        ret.data = data_bioem
        ret.feature_names = feature_names_bioem
    
    if(data_type == "archea"):
        ret.data = data_archea
        ret.feature_names = feature_names_archea
    
    if(data_type == "all_relative"):
        ret.data = data_allr
        ret.feature_names = feature_names_allr

    # [print(x) for x in ret.target]
    # exit()

    ret.data = np.array(ret.data,dtype=np.float)
    if merge_features:
        new_index = {}
        new_data = np.zeros((ret.data.shape[0],
                             len(set(ret.feature_names))),
                            dtype=np.float)
        new_feature_names = []
        for i,fname in enumerate(ret.feature_names):
            if not (fname in new_index):
                new_index[fname] = len(new_index)
                new_feature_names.append(fname)
            new_data[:,new_index[fname]] = new_data[:,new_index[fname]]+ret.data[:,i]
        

        ret.feature_names = np.array(new_feature_names)
        # print(ret.data)
        ret.data = new_data
        # print([x for x in ret.data[ret.data != 0]])
        # exit()
        # print(ret.data[:,ret.feature_names=="Zoogloea"])
        # print(ret.data[:,ret.feature_names=="Z20"])
        # print(ret.feature_names)

    mask = np.zeros(len(ret.feature_names)) == 0
    for bw in ["Ambiguous_taxa", "Other", "Unknown"]:
        for i,ft_name in enumerate(ret.feature_names):
            if bw in ft_name:
                mask[i] = False
    ret.data = ret.data[:,mask]
    ret.feature_names = ret.feature_names[mask]

    if abundance_limiar < 1.0:

        abd = np.sum(ret.data, axis=0) # abundancies
        s_abd = sorted(abd) # abundancies
        
        idx = int(np.round(abundance_limiar*len(s_abd)))
        limiar = s_abd[-idx]
        mask = abd >= limiar
        
        # ret.name = np.array(ret.name)[:, mask]
        # ret.target_names = np.array(ret.target_names)[:, mask]
        # print(ret.target_levels)
        
        ret.feature_names = np.array(ret.feature_names)[mask]
        ret.data = np.array(ret.data)[:, mask]
        # if(data_type == "all_relative"):
        #     ret.target_levels = {x:np.array(ret.target_levels[x])[mask] for x in ret.target_levels
        #                          if x in ret.feature_names}
    
    
    

    return ret
    

if __name__=="__main__":
    data =load_biogas(data_type="bacteria", label_type="grouped")
    # print(data.data.shape)
    # for i in range(data.data.shape[0]):
    #     data.data[i,:] = data.data[i,:]/sum(data.data[i,:])
    
    # i = list(data.feature_names).index("vadinBC27 wastewater-sludge group")
    # print(list(zip(data.label_encoder.inverse_transform(data.target),data.data[:,i])))
    # d1 = data.data[:,i]

    data = load_biogas(data_type="all_relative", label_type="grouped",
                      relative_taxa_level=5, merge_features=True,
                      filter_by_taxa_level=(0,"Bacteria"))
    idx = np.where(data.feature_names=="Neochlamydia")[0]
    print(','.join(['%f' % x[0] for x in data.data[:, idx]]))
    idx = np.where(data.feature_names=="uncultured Parcubacteria group bacterium")[0]
    print(','.join(['%f' %x[0] for x in data.data[:, idx]]))
    # i = list(data.feature_names).index("D_0__BacteriaD_1__BacteroidetesD_2__BacteroidiaD_3__BacteroidalesD_4__RikenellaceaeD_5__vadinBC27 wastewater-sludge group")
    # print(list(data.data[:,i]))
    # print(list(zip(data.label_encoder.inverse_transform(data.target),data.data[:,i])))
    # print(sum(data.data[:,i]))
    # d2 = data.data[:,i]

    # print(sorted(d1/np.sum(d1)))
    # print(sorted(d2/np.sum(d2)))

    # l = [x.split("D_4__")[1] for x in data.feature_names if "D_4__" in x]
    # l = [x.split("D_5__")[0] for x in l if "D_5__" in x]
    # [print(x) for x in set(l) if "vadin" in x or "Vadin" in x]
    # print(len(set(l)))
    # print(data.target_levels[0])
    # print(data.target_levels[1])
    # print(data.target_levels[2])
    # print(data.target_levels[0]=="Bacteria")
    # print(len([x for x in data.target_levels[0] if "Archaea" in x]))
    # print(len(data.feature_names))
    # print(data.feature_names[data.target_levels[0]=="Archaea"])
    data = load_biogas(data_type = "all_relative", label_type="grouped", label_value="name", relative_taxa_level=5)
    print(data.data.shape)
    
    data = load_biogas(data_type = "all_relative", label_type="binary", relative_taxa_level=5)
    print(data.data.shape)