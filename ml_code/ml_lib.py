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
from sklearn.cluster import AgglomerativeClustering

# Agglomerative Graph
import seaborn as sns; sns.set(color_codes=True) 
from scipy.spatial import distance
from scipy.cluster import hierarchy

# Spectral
from sklearn.cluster import SpectralClustering

# Cluster analysis
from sklearn import metrics

# Dimensionality Reduction
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap

class data_proc():

    def convertTableToList(tab):
        return tab.flatten().tolist()

    def convertArrayToDataFrame(array, rowLabels):
        return pd.DataFrame(data=array,
                            index=rowLabels)

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

    def agglomerativeClustering(data, k):
        cluster = AgglomerativeClustering(n_clusters=k, 
                                          affinity='euclidean', 
                                          linkage='ward')
        y_pred = cluster.fit_predict(data)
        return y_pred


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
        rand = metrics.adjusted_rand_score(labels_actual, labels_predicted)
        # consensus measure: 1.0 is perfect match, independent labels have negative values
        miscore = metrics.adjusted_mutual_info_score(labels_actual, labels_predicted, average_method='arithmetic')
        # homogeneity = each cluster contains only members of a single class [0.0, 1.0]
        # symmetric, [0.0, 1.0]
        hscore = metrics.homogeneity_score(labels_actual, labels_predicted)
        # completeness = all members of a given class are assigned to the same cluster [0.0, 1.0]
        # symmetric, [0.0, 1.0]
        cscore = metrics.completeness_score(labels_actual, labels_predicted)
        # harmonic mean of homogeneity and completeness: 
        # beta = 1 for equal
        # beta > 1 for more weight on completeness
        # beta < 1 for more weight on homogeneity
        # symmetric, [0.0, 1.0]
        vscore = metrics.v_measure_score(labels_actual, labels_predicted)

        return [rand, miscore, hscore, cscore, vscore]

    def analyzeClusteringModel(data, labels):
        # silhouette coefficient: measure of mean distance between sample and rest of its class and mean distance between sample and points in nearest cluster
        # range -1 for incorrect clustering to 1 for correct clustering
        # biased toward higher scores for convex clusters
        sscore = metrics.silhouette_score(data, labels)
        # calinski-harabasz index: within cluster dispersion vs. between-cluster dispersion
        # higher score means dense, well separated clusters
        chscore = metrics.calinski_harabasz_score(data, labels)
        # davies-bouldin index
        # lower index  = better separation between clusters
        # drawback: higher for convex clusters, low values not necessarily related to the best information retrieval
        dbscore = metrics.davies_bouldin_score(data, labels)

        return [sscore, chscore, dbscore]

class supervised():

    def regression():
        pass

    def svm():
        pass

    def generateNN(): # lstm neural network
        pass

class dim_redux():

    def performPCA(data, ndim=2):
        pca = PCA(n_components=ndim)
        components = pca.fit_transform(data)
        return components

    def performTSNE(data, ndim=2):
        tsne = TSNE(n_components=ndim,
                    init='random', 
                    random_state=0,
                    method='exact')
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

    def showClusterResults(features, classes, title="", fn=""):
        fig = plt.figure()
        plt.scatter(features[:, 0], features[:, 1], c=classes, cmap='jet', alpha=.7)
        plt.title(title)
        plt.show()

        if not fn == "":
            fig.savefig("figures/"+fn, bbox_inches='tight')

    def graphAgglomerativeClustering(data, classes, sitesLookup):
        current_palette = sns.color_palette("bright")
        colors=[current_palette[sitesLookup[c]] for c in classes]
        legend = [mpatches.Patch(color=current_palette[sitesLookup[c]], label=c) for c in list(set(classes))]
        # sns.clustermap # https://seaborn.pydata.org/generated/seaborn.clustermap.html
        # cosine distance # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
        # linkage? # https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
        row_linkage = hierarchy.linkage(distance.pdist(data), method='average')
        g = sns.clustermap(data,
                           row_linkage=row_linkage,
                           col_cluster=False,
                           row_colors=colors,
                           cmap="jet") #, metric='cosine')
        
        l = g.ax_heatmap.legend(loc='upper right', handles=legend, frameon=True)
        l.set_title(title='Site Key')

        # return the calculated dendrogram
        return row_linkage

    def plotScores(scores, title="", legendLabels=None, fn=""):
        signals = np.asarray([np.asarray(k) for k in scores])
        print(signals.shape) # should be the number of scores by the number of k
        # make the plot
        colormap = sns.color_palette("bright")
        sns.set_palette(colormap)
        fig = plt.figure(1) #figsize=(13, 7))
        ax =fig.add_subplot(111)
        ax.plot(signals, '.-')
        ax.set_title(title)
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("Score Value")
        
        if fn is not "":
            if legendLabels is not None:
                legend = ax.legend(legendLabels, frameon=True, bbox_to_anchor=(1.04,.5), loc="center left")
                frame = legend.get_frame()
                frame.set_facecolor('white')

                print(fn)
                fig.savefig("figures/"+fn, bbox_extra_artists=(legend,), bbox_inches='tight')
            
            else:
                print(fn)
                fig.savefig("figures/"+fn)
            
        plt.show()

def performUnsupervisedKAnalysis(data, y_actual, titleFeats="", fnFeats="allfeats_fd"):
    # rand, mutual info score, homogeneity, completeness, vscore
    # silhouette score, calinski harabasz score, davies bouldin score

    # Grade the cluster labels
    resultsScoresSitesK = []
    resultsScoresSitesS = []
    resultsScoresSitesA = []
    # Grade the model itself
    modelScoresK = []
    modelScoresS = []
    modelScoresA = []

    for k in range(2, 40):
        # K-means
        y_kmeans = unsupervised.kmeansClustering(data, k)
        resultsScoresSitesK.append(unsupervised.analyzeClusteringResults(y_actual, y_kmeans[0]))
        modelScoresK.append(unsupervised.analyzeClusteringModel(data, y_kmeans[0]))

        # Spectral
        try:
            y_spectral = unsupervised.spectralClustering(data, k)
            resultsScoresSitesS.append(unsupervised.analyzeClusteringResults(y_actual, y_spectral))
            modelScoresS.append(unsupervised.analyzeClusteringModel(data, y_spectral))
        except:
            resultsScoresSitesS.append([-10, -10, -10])
            modelScoresS.append([-10, -10, -10])
            
        # Agglomerative
        y_agg = unsupervised.agglomerativeClustering(data, k)
        resultsScoresSitesA.append(unsupervised.analyzeClusteringResults(y_actual, y_agg))
        modelScoresA.append(unsupervised.analyzeClusteringModel(data, y_agg))

    labels_resultsScore = ['Adjusted Rand Score', 'Adjusted Mutual Information', 'Homogeneity', 'Completeness', 'Balanced V-Measure Score']
    labels_modelsScore = ['Silhouette Score', 'Calinski-Harabasz Score', 'Davies-Bouldin Score']

    # Plot the cluster label scores
    visualization.plotScores(resultsScoresSitesK, 
                             "K-means Clustering: Cluster Scores ("+titleFeats+")", 
                             labels_resultsScore, 
                             fn='clustering_figs_kmeans_'+fnFeats+'_cluster_scores.png')

    visualization.plotScores(resultsScoresSitesS, 
                             "Spectral Clustering: Cluster Scores ("+titleFeats+")", 
                             labels_resultsScore, 
                             fn='clustering_figs_spectral_'+fnFeats+'_cluster_scores.png')

    visualization.plotScores(resultsScoresSitesA, 
                             "Agglomerative Clustering: Cluster Scores ("+titleFeats+")", 
                             labels_resultsScore, 
                             fn='clustering_figs_agg_'+fnFeats+'_cluster_scores.png')
    # Plot the model scores
    visualization.plotScores(modelScoresK, 
                             "K-means Clustering: Model Scores ("+titleFeats+")",
                              labels_modelsScore, 
                              fn='clustering_figs_kmeans_'+fnFeats+'_model_scores.png')

    visualization.plotScores(modelScoresS,
                             "Spectral Clustering: Model Scores ("+titleFeats+")", 
                             labels_modelsScore, 
                             fn='clustering_figs_spectral_'+fnFeats+'_model_scores.png')
    visualization.plotScores(modelScoresA,
                             "Agglomerative Clustering: Model Scores ("+titleFeats+")", 
                             labels_modelsScore, 
                             fn='clustering_figs_agg_'+fnFeats+'_model_scores.png')
