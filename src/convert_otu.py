from external_dataset import load_biogas

try:
    import numpy as np
except:
    import _numpypy as np

# db = load_biogas(data_type="bacteria", label_type="grouped", label_value='name')
# db = load_biogas(data_type="archea", label_type="grouped", label_value='name')

label_value="name"

# biogas_database_relative_grouped_0 = load_biogas(data_type = "all_relative", label_type="grouped", label_value=label_value, relative_taxa_level=0)
# biogas_database_relative_grouped_1 = load_biogas(data_type = "all_relative", label_type="grouped", label_value=label_value, relative_taxa_level=1)
# biogas_database_relative_grouped_2 = load_biogas(data_type = "all_relative", label_type="grouped", label_value=label_value, relative_taxa_level=2)
# biogas_database_relative_grouped_3 = load_biogas(data_type = "all_relative", label_type="grouped", label_value=label_value, relative_taxa_level=3)
# biogas_database_relative_grouped_4 = load_biogas(data_type = "all_relative", label_type="grouped", label_value=label_value, relative_taxa_level=4)

biogas_database_relative_grouped_5 = load_biogas(data_type = "all_relative", label_type="grouped", label_value=label_value, relative_taxa_level=5)
biogas_database_relative_grouped_5_arch = load_biogas(data_type = "all_relative", label_type="grouped", label_value=label_value, relative_taxa_level=5, filter_by_taxa_level=(0,"Archaea"))
biogas_database_relative_grouped_5_bact = load_biogas(data_type = "all_relative", label_type="grouped", label_value=label_value, relative_taxa_level=5, filter_by_taxa_level=(0,"Bacteria"))

# biogas_database_relative_grouped_0_fill = load_biogas(data_type = "all_relative", label_type="grouped", label_value=label_value, relative_taxa_level=0,abundance_limiar=0.05)
# biogas_database_relative_grouped_1_fill = load_biogas(data_type = "all_relative", label_type="grouped", label_value=label_value, relative_taxa_level=1,abundance_limiar=0.05)
# biogas_database_relative_grouped_2_fill = load_biogas(data_type = "all_relative", label_type="grouped", label_value=label_value, relative_taxa_level=2,abundance_limiar=0.05)
# biogas_database_relative_grouped_3_fill = load_biogas(data_type = "all_relative", label_type="grouped", label_value=label_value, relative_taxa_level=3,abundance_limiar=0.05)
# biogas_database_relative_grouped_4_fill = load_biogas(data_type = "all_relative", label_type="grouped", label_value=label_value, relative_taxa_level=4,abundance_limiar=0.05)

biogas_database_relative_grouped_5_fill = load_biogas(data_type = "all_relative", label_type="grouped", label_value=label_value, relative_taxa_level=5,abundance_limiar=0.05)
# biogas_database_relative_grouped_5_arch_fill = load_biogas(data_type = "all_relative", label_type="grouped", label_value=label_value, relative_taxa_level=5,abundance_limiar=0.05, filter_by_taxa_level=(0,"Archaea"))
biogas_database_relative_grouped_5_bact_fill = load_biogas(data_type = "all_relative", label_type="grouped", label_value=label_value, relative_taxa_level=5,abundance_limiar=0.05, filter_by_taxa_level=(0,"Bacteria"))


databases = [
    ("all_relative", biogas_database_relative_grouped_5),
    ("arch_relative", biogas_database_relative_grouped_5_arch),
    ("bact_relative", biogas_database_relative_grouped_5_bact),
    ("all_relative_fill", biogas_database_relative_grouped_5_fill),
    # ("arch_relative_fill", biogas_database_relative_grouped_5_arch_fill),
    ("bact_relative_fill", biogas_database_relative_grouped_5_bact_fill),
]


for db_name, db in databases:
    file_text = ""
    dataX, dataY = (db.data, db.target)

    # header = "OTU_id\t"+'\t'.join(list(range(len(dataX))))

    sample_ids = [str(x) for x in range(len(dataX))]

    # header
    file_text+="OTU_id\t"+'\t'.join(sample_ids)+"\n"

    for i,otu in enumerate(dataX.T):
        string_list = [str(x) for x in otu]
        file_text+=str(i)+"\t"+'\t'.join(string_list)+"\n"
    

    with open(db_name+".txt", 'w') as fp:
        fp.write(file_text)
        fp.close()
    

