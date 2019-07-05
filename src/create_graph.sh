echo "Computing scores for each OTU..."
# python3 otu_importance.py --normalize
python3 otu_importance.py


regression_limiar_alpha=0.8
regression_limiar_beta=0.95

echo "Computing correlations for all entries with linear regression correlation..."
python3 otu_correlation.py --dtype all_relative --json_name all --label_type grouped --taxa_level 5 --limiar_alpha $regression_limiar_alpha --score_type reg
echo "Computing correlations for all entries with pearson correlation..."
python3 otu_correlation.py --dtype all_relative --json_name all_pearson --label_type grouped --taxa_level 5 --limiar_alpha $regression_limiar_alpha --score_type pearson
echo "Computing correlations for all entries with spearman correlation..."
python3 otu_correlation.py --dtype all_relative --json_name all_spearman --label_type grouped --taxa_level 5 --limiar_alpha $regression_limiar_alpha --score_type spearman
echo "Computing correlations for all entries with kendall correlation..."
python3 otu_correlation.py --dtype all_relative --json_name all_kendall --label_type grouped --taxa_level 5 --limiar_alpha $regression_limiar_alpha --score_type kendall

echo "Computing correlations for Archaea entries with linear regression correlation..."
python3 otu_correlation.py --dtype all_relative --json_name arch --label_type grouped --taxa_level 5 --filter_by_taxa_level 0 --filter_by_taxa_name Archaea --limiar_alpha $regression_limiar_alpha --limiar_beta 0.5 --score_type reg
echo "Computing correlations for Archaea entries with pearson correlation..."
python3 otu_correlation.py --dtype all_relative --json_name arch_pearson --label_type grouped --taxa_level 5 --filter_by_taxa_level 0 --filter_by_taxa_name Archaea --limiar_alpha $regression_limiar_alpha --limiar_beta 0.5 --score_type pearson
echo "Computing correlations for Archaea entries with spearman correlation..."
python3 otu_correlation.py --dtype all_relative --json_name arch_spearman --label_type grouped --taxa_level 5 --filter_by_taxa_level 0 --filter_by_taxa_name Archaea --limiar_alpha $regression_limiar_alpha --limiar_beta 0.5 --score_type spearman
echo "Computing correlations for Archaea entries with kendall correlation..."
python3 otu_correlation.py --dtype all_relative --json_name arch_kendall --label_type grouped --taxa_level 5 --filter_by_taxa_level 0 --filter_by_taxa_name Archaea --limiar_alpha $regression_limiar_alpha --limiar_beta 0.5 --score_type kendall

echo "Computing correlations for Bacteria entries with linear regression correlation..."
python3 otu_correlation.py --dtype all_relative --json_name bact --label_type grouped --taxa_level 5 --filter_by_taxa_level 0 --filter_by_taxa_name Bacteria --limiar_alpha $regression_limiar_alpha --score_type reg
echo "Computing correlations for Bacteria entries with pearson correlation..."
python3 otu_correlation.py --dtype all_relative --json_name bact_pearson --label_type grouped --taxa_level 5 --filter_by_taxa_level 0 --filter_by_taxa_name Bacteria --limiar_alpha $regression_limiar_alpha --score_type pearson
echo "Computing correlations for Bacteria entries with spearman correlation..."
python3 otu_correlation.py --dtype all_relative --json_name bact_spearman --label_type grouped --taxa_level 5 --filter_by_taxa_level 0 --filter_by_taxa_name Bacteria --limiar_alpha $regression_limiar_alpha --score_type spearman
echo "Computing correlations for Bacteria entries with kendall correlation..."
python3 otu_correlation.py --dtype all_relative --json_name bact_kendall --label_type grouped --taxa_level 5 --filter_by_taxa_level 0 --filter_by_taxa_name Bacteria --limiar_alpha $regression_limiar_alpha --score_type kendall

echo "Computing correlations for all entries (5% most abundant) with linear regression correlation..."
python3 otu_correlation.py --dtype all_relative --json_name all_fill --label_type grouped --taxa_level 5 --abundance_limiar 0.05 --limiar_alpha $regression_limiar_alpha --score_type reg
echo "Computing correlations for all entries (5% most abundant) with pearson correlation..."
python3 otu_correlation.py --dtype all_relative --json_name all_fill_pearson --label_type grouped --taxa_level 5 --abundance_limiar 0.05 --limiar_alpha $regression_limiar_alpha --score_type pearson
echo "Computing correlations for all entries (5% most abundant) with spearman correlation..."
python3 otu_correlation.py --dtype all_relative --json_name all_fill_spearman --label_type grouped --taxa_level 5 --abundance_limiar 0.05 --limiar_alpha $regression_limiar_alpha --score_type spearman
echo "Computing correlations for all entries (5% most abundant) with kendall correlation..."
python3 otu_correlation.py --dtype all_relative --json_name all_fill_kendall --label_type grouped --taxa_level 5 --abundance_limiar 0.05 --limiar_alpha $regression_limiar_alpha --score_type kendall

echo "Computing correlations for Bacteria (5% most abundant) entries with linear regression correlation..."
python3 otu_correlation.py --dtype all_relative --json_name bact_fill --label_type grouped --taxa_level 5 --abundance_limiar 0.05 --filter_by_taxa_level 0 --filter_by_taxa_name Bacteria --limiar_alpha $regression_limiar_alpha --score_type reg
echo "Computing correlations for Bacteria (5% most abundant) entries with pearson correlation..."
python3 otu_correlation.py --dtype all_relative --json_name bact_fill_pearson --label_type grouped --taxa_level 5 --abundance_limiar 0.05 --filter_by_taxa_level 0 --filter_by_taxa_name Bacteria --limiar_alpha $regression_limiar_alpha --score_type pearson
echo "Computing correlations for Bacteria (5% most abundant) entries with spearman correlation..."
python3 otu_correlation.py --dtype all_relative --json_name bact_fill_spearman --label_type grouped --taxa_level 5 --abundance_limiar 0.05 --filter_by_taxa_level 0 --filter_by_taxa_name Bacteria --limiar_alpha $regression_limiar_alpha --score_type spearman
echo "Computing correlations for Bacteria (5% most abundant) entries with kendall correlation..."
python3 otu_correlation.py --dtype all_relative --json_name bact_fill_kendall --label_type grouped --taxa_level 5 --abundance_limiar 0.05 --filter_by_taxa_level 0 --filter_by_taxa_name Bacteria --limiar_alpha $regression_limiar_alpha --score_type kendall

python3 create_js_from_json.py