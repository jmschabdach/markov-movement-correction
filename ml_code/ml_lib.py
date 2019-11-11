# Imports here
import pandas as pd
import numpy as np
import os

# Data filtering
import scipy.stats

# Plotting
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches

# Kmeans
import sklearn
from sklearn.cluster import KMeans

# Agglomerative
import seaborn as sns; sns.set(color_codes=True) 

# Spectral
from sklearn.cluster import SpectralClustering

# Cluster analysis
from sklearn import metrics

# Dimensionality Reduction
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap

class data_proc():
    # Perform outlier detection: identify sequences where there is an outlier in any of the metrics. Don't use these sequences.
    # Using the t-statistic: like a z-score, but only the sample mean and std. are known
    # Except in python using scipy, it looks like the zscore is the way to go
    # Note: outliers are specified as DAG registered sequences who have a zscore at any frame outside of 3 std above and below the mean

    def identifyOutliers(dataDf):
        # Calculate the zscore for all of the columns
        zscores = (dataDf - dataDf.mean())/dataDf.std()
        zscores = zscores.fillna(0)
    #     plt.hist(zscores.values.flatten())
        
        highMotionSubjects = []
        
        # For each row in the data frame of zscores
        for index, row in zscores.iterrows():
            # If the row contains a value > 3 or < -3 (value is over 3 standard deviations from the mean)
            if min(row) < -3 or max(row) > 3:
                # Add the label for that row to a list (print for now)
    #             print(index)
                highMotionSubjects.append(index)
                
        return highMotionSubjects

class unsupervised():

    def agglomerativeClustering(data, classes, sitesLookup):
        current_palette = sns.color_palette("bright")
        colors=[current_palette[sitesLookup[c]] for c in classes]
        legend = [mpatches.Patch(color=current_palette[sitesLookup[c]], label=c) for c in list(set(classes))]
        # sns.clustermap # https://seaborn.pydata.org/generated/seaborn.clustermap.html
        # cosine distance # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
        # linkage? # https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
        g = sns.clustermap(data,
                           col_cluster=False,
                           row_colors=colors,
                           cmap="jet") #, metric='cosine')
        
        l = g.ax_heatmap.legend(loc='upper right', handles=legend, frameon=True)
        l.set_title(title='Site Key')

    def kmeansClustering(data, k):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(data)
        y_pred = kmeans.predict(data)
        # Need to analyze results: how many subjects from each class are in each cluster
        return y_pred, kmeans

    def spectralClustering(data, k):
        spectral=SpectralClustering(n_clusters=k)
        y_pred = spectral.fit_predict(data)
        return y_pred

    # https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation
    def analyzeClusteringResults(labels_actual, labels_predicted):
        # consensus measure: 1.0 is perfect match, close to 0.0 or negative is bad
        metrics.adjusted_rand_score(labels_actual, labels_predicted)
        # consensus measure: 1.0 is perfect match, independent labels have negative values
        metrics.adjusted_mutual_info_score(labels_actual, labels_predicted)
        # homogeneity = each cluster contains only members of a single class [0.0, 1.0]
        # symmetric, [0.0, 1.0]
        metrics.homogeneity_score(labels_actual, labels_predicted)
        # completeness = all members of a given class are assigned to the same cluster [0.0, 1.0]
        # symmetric, [0.0, 1.0]
        metrics.completeness_score(labels_actual, labels_predicted)
        # harmonic mean of homogeneity and completeness: 
        # beta = 1 for equal
        # beta > 1 for more weight on completeness
        # beta < 1 for more weight on homogeneity
        # symmetric, [0.0, 1.0]
        metrics.v_measure_score(labels_actual, labels_predicted)

    def analyzeClusteringModel(data, labels):
        # silhouette coefficient: measure of mean distance between sample and rest of its class and mean distance between sample and points in nearest cluster
        # range -1 for incorrect clustering to 1 for correct clustering
        # biased toward higher scores for convex clusters
        metrics.silhouette_score(data, labels)
        # calinski-harabasz index: within cluster dispersion vs. between-cluster dispersion
        # higher score means dense, well separated clusters
        metrics.calinski_harabasz_score(data, labels)
        # davies-bouldin index
        # lower index  = better separation between clusters
        # drawback: higher for convex clusters, low values not necessarily related to the best information retrieval
        metrics.davies_bouldin_score(data, labels)

class supervised():

    def regression():
        pass

    def svm():
        pass

    def generateNN(): # lstm neural network
        pass

class dim_redux():

    def performPCA(data):
        pca = PCA(n_components=2)
        components = pca.fit_transform(data)
        return components

    def performTSNE(data):
        tsne = TSNE(n_components=2,
                    init='random', 
                    random_state=0)
        tsne_projected = tsne.fit_transform(data)
        return tsne_projected

    def performUMAP(data):
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(data)
        return embedding



class visualization():

    def viewDataList(y, yax="", title=""):
        fig = plt.figure()
        plt.plot(y)
        plt.xlabel("Image Volume Number")
        plt.ylabel(yax)
        plt.title(title)
    
    def viewDataArray(data):
        sns.heatmap(data)

    def showClusterResults(features, classes, title=""):
        fig = plt.figure()
        plt.scatter(features[:, 0], features[:, 1], c=classes, cmap='viridis')
        plt.title(title)
        plt.show()