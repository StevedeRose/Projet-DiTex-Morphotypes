# [2021-M2] ***DiTex - Morphotypes***
DiTeX is a joint research and development laboratory between the University of Technology of Troyes and the French Institute of Textiles and Clothing.

## Project's Presentation
This project is part of the Master in Scientific Computing and Mathematics for Information.
One of the objectives of this master is to provide its students with advanced skills in data analysis.
This work is an excellent opportunity to put into practice what has been learned throughout the master's degree.

## Introduction
The variety of human morphologies is an important issue for the textile-apparel industry.
Indeed, sizing systems currently used by companies have to be continuously updated or adapted to the population target.

For this reason, the Textile-Apparel-Industry requires a very accurate sizing system to minimize their costs and satisfy their customers.
However, the specific constraints of human morphotologies complicate the sizing system definition procedure and distributors prefer to use standard sizing system rather than an intelligent system suitable to their customers.

Until now, the morphotypes of a population are extracted from measurement charts.
However, new technologies such as 3D body scanning open new opportunities to enhance the morphotype generation from a sample of population especially with the 3D data of bodies.

The aim of this research is to define an exhaustive methodology to obtain a clustering of human morphology shapes representative of a population and to extract the most significant morphotype of each class.
Clustering methods are implemented and the performances are evaluated using real data.

## How to replicate the results.
The easiest way is to download one of the notebooks and run it.  
You can run them in Google Colaboratory.

You can also reproduce it yourself with these Python codes:  
Necessary packages: `pandas`, `numpy`, `seaborn`, `scipy`, `scikit-learn`, `matplotlib`, `scikit-learn-extra` and `yellowbrick`
### Import of the necessary packages and functions.
```py
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.cluster.hierarchy as shc
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from yellowbrick.cluster import KElbowVisualizer
```
### Importing the ANSUR II database.
#### Female data set
```py
df = pd.read_csv('http://tools.openlab.psu.edu/publicData/ANSUR_II_FEMALE_Public.csv')
```
#### Male data set
```py
df = pd.read_csv('http://tools.openlab.psu.edu/publicData/ANSUR_II_MALE_Public.csv', encoding = "ISO-8859-1")
```
#### Distribution of missing values
```py
plt.figure(figsize=(15, 9))
sns.heatmap(df.isnull(), cmap='viridis')
plt.show()
```
#### Correlation table of characteristics
```py
fig, ax = plt.subplots(figsize=(14, 12)) # Pour augmenter la taille de la figure
sns.heatmap(df.corr(), cmap="jet")
plt.show()
```
### Extraction of database columns.
```py
columns = ['bicristalbreadth',
           'buttockcircumference',
           'buttockdepth',
           'chestbreadth',
           'chestcircumference',
           'chestdepth',
           'hipbreadth',
           'lowerthighcircumference',
           'shouldercircumference',
           'thighcircumference',
           'verticaltrunkcircumferenceusa',
           'waistbreadth',
           'waistcircumference',
           'waistdepth']

df_fit_ = pd.DataFrame(df, columns=columns)
```
### Correlation matrix of the 14 torso and thigh measurements
```py
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(df_fit_.corr(), cmap="jet")
plt.show()
```
### Data normalisation
```py
df_fit = (df_fit_.copy() - df_fit_.mean()) / df_fit_.std()
```
### Elbow Method
```py
model = KMedoids(method='pam') # AgglomerativeClustering(linkage='ward')
visualizer = KElbowVisualizer(model, k=(2, 16), timings=False)

visualizer.fit(df_fit)   # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure
```
### Visualisation
#### Function
```py
colors = ['ro', 'g^', 'bs', 'cX', 'yP', 'm*', 'kH']


def clustering(data, method, plot=True):
    method.fit(data)
    identified_clusters = method.fit_predict(data)

    acp = PCA(n_components=2, svd_solver='full')
    coord = acp.fit_transform(data)

    if not plot:
        print(100*sum(acp.explained_variance_ratio_),
              '% ot the variance retained')
    else:
        x_proj = acp.transform(data)
        x_proj = pd.DataFrame(data=x_proj, columns=[
            'Composante principale 1', 'Composante principale 2'])
        x_proj['Clusters'] = identified_clusters

        fig, ax = plt.subplots(figsize=(25, 25))
        plt.xlabel('First component: {:.2f}%'.format(
            100*acp.explained_variance_ratio_[0]), fontsize=20)
        plt.ylabel('Second component: {:.2f}%'.format(
            100*acp.explained_variance_ratio_[1]), fontsize=20)
        for i in range(x_proj['Clusters'].max()+1):
            plt.plot(x_proj[x_proj['Clusters'] == i]['Composante principale 1'],
                     x_proj[x_proj['Clusters'] == i]['Composante principale 2'],
                     colors[i], label='Cluster {}'.format(i+1), alpha=0.7)
        plt.legend(loc='best', fontsize=20)
        plt.show()
```
#### Usage
- K-Medoids
```py
clustering(df_fit, KMedoids(n_clusters=6, method='pam'))
```
- Ward's Method
```py
clustering(df_fit, AgglomerativeClustering(n_clusters=4, linkage='ward'))
```
#### Dendrogram
```py
plt.figure(figsize=(12, 6))
plt.title("Body Shape Dendrogram")
dend = shc.dendrogram(shc.linkage(df_fit, method='ward'),
                      p=50, truncate_mode='lastp', show_leaf_counts=False)
plt.tight_layout()
plt.show()
```
### Clusters Descriptions.
#### Data preparation
```py
method = KMedoids(n_clusters=6, method='pam') # AgglomerativeClustering(n_clusters=4, linkage='ward')
method.fit(df_fit)
identified_clusters = method.fit_predict(df_fit)

df_means = df.select_dtypes(include='number').copy()

df_means["Cluster"] = identified_clusters.copy()
df_means["Weight"] = df_means["Weightlbs"].copy() * 0.453592
df_means["Height"] = df_means["Heightin"].copy() * 2.54
df_means["BMI"] = 10000 * df_means['Weight'] / df_means['Height']**2
df_fit_["Cluster"] = identified_clusters.copy()

todrop = ['subjectid',
          'SubjectNumericRace',
          'DODRace',
          'Weightlbs',
          'weightkg',
          'Heightin']

for col in todrop:
    df_means.pop(col)

df_medoids = df_fit.copy()
df_medoids["Cluster"] = identified_clusters.copy()
```
#### Data set description
Use the pandas dataframe `describe` method
```py
df_means.describe()
```
#### Cluster description
For the *n*-th cluster
```py
cluster2 = df_means[df_means["Cluster"]==1]
cluster2.pop('Cluster')
cluster2.describe()
```
#### Finding a cluster medoid index 
For the *n*-th cluster
- Euclidean distance
```py
df_medoid_n = df_medoids[df_medoids["Cluster"] == n-1]
medoid_n = np.argmin(np.sqrt(((df_medoid_n - df_medoid_n.mean())**2).sum(axis=1)))
```
- Squared Euclidean distance
```py
df_medoid_n = df_medoids[df_medoids["Cluster"] == n-1]
medoid_n = np.argmin(((df_medoid_n - df_medoid_n.mean())**2).sum(axis=1))
```
- Medoid's full measurements
```py
df_fit_[df_fit_["Cluster"] == n-1].iloc[medoid_n]
```
