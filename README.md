Here is a breif explaination of all the code in this repository, 

50%KSstat.py = KS statistics worked out for each feature, taking a random 50% of the features data, and comparing its distribution to the other half, a number of times. 

Dash app.py = Currently in development, a dash application that allows for the loading of the UMAP data in json format, and then subseqeunt highlighting and extraction of a speficies area of data points. 

DataSplitter.py = A script that splits the original dataset into hand selected feature groups. 

Disttribution of Variables.r = A R script to work out multiple distribution metrics of each feature. 

Feature removal.py = Takes a list of features, i.e. the least contribting variables from a PCA analysis, and removes them from the original dataset, to create a dataset of features with the largest proportion of group variation. 

Merging of T and M.py = Adds the appropriate treatment and coating condition from the MAcsExpt1 Key.xml file to their respective samples in the full macs dataset. 

PCA.py = Standard PCA conducted on the Macs dataset, which filters for the top contributions in the top 20 components and saves them to a csv file. 

UMAP analysis of IRIS.py = UMAP of IRIS dataset (old) 

UMAP for wine quality.py = UMAP of Wine quality dataset (old) 

UMAP standardisations.py = Going over numerous standardisation techniques automatically for the Macs Dataset. 

UMAPx3DxAvg.py = Umap analysis which finds the average points of each identifier, and vis in 3D, and configurable dist and num of neighbours and metric.  

UMAPxAvgxMdata.py = UMAP with all of the feature groups from the DataSplitter.py, with average identifier points and 3D vis. 

UMAPxTargetx2D.py = UMAP with unsupervised learning to target clustering in the Identifer feature, in 2D. 

UMAPxspread.py = UMAP analysis in 3D with configurable spread options. 

Anova.py = Anova tests for all the features in the Macs dataset.

(new) ESF.py = Elasticnet Feature selection and dataset merger. 
