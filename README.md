# [2021-M2] DiTex — Human Body Morphotype Classification

> **DiTeX** is a joint research and development laboratory between the [University of Technology of Troyes (UTT)](https://www.utt.fr/) and the [French Institute of Textiles and Clothing (IFTH)](https://www.ifth.org/).

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Background & Motivation](#background--motivation)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Feature Selection](#feature-selection)
  - [Data Preprocessing](#data-preprocessing)
  - [Clustering Algorithms](#clustering-algorithms)
  - [Cluster Evaluation](#cluster-evaluation)
  - [Morphotype Extraction](#morphotype-extraction)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [Running the Notebooks](#running-the-notebooks)
- [Step-by-Step Code Walkthrough](#step-by-step-code-walkthrough)
- [References](#references)
- [Authors](#authors)

---

## Project Overview

This project is part of the **Master in Scientific Computing and Mathematics for Information** at UTT. Its goal is to design and evaluate an end-to-end methodology for clustering human body morphologies and extracting representative morphotypes from real 3D anthropometric data.

The work aims to support the textile-apparel industry in building smarter, data-driven sizing systems that better represent the diversity of human body shapes.

---

## Background & Motivation

The diversity of human body shapes is a critical challenge for the textile-apparel industry. Sizing systems must continuously evolve to reflect actual population morphologies, yet most companies still rely on standard sizing tables that fail to capture this diversity.

Key drivers of this project:
- **Industry need**: Inaccurate sizing leads to customer dissatisfaction and increased production waste.
- **Technological opportunity**: 3D body scanning now enables the capture of detailed body shape data at scale, going far beyond traditional tape-measure surveys.
- **Research gap**: Most existing sizing systems are derived from simple 1D measurement charts; there is a need for methods that leverage richer shape descriptors.

This research defines an **exhaustive methodology** to:
1. Cluster human morphologies from 3D anthropometric measurements.
2. Evaluate the quality of the resulting clusters.
3. Extract a representative **morphotype** (medoid) for each cluster.

---

## Dataset

The project uses the publicly available **ANSUR II** (Anthropometric Survey of US Army Personnel, 2012) database, collected by the US Army Natick Soldier Research, Development & Engineering Center.

| Property | Details |
|---|---|
| **Female subjects** | 1,986 participants |
| **Male subjects** | 4,082 participants |
| **Measurements per subject** | 93 anthropometric variables |
| **Additional variables** | Demographics, weight, height, BMI |
| **Access** | [ANSUR II Public Data](http://tools.openlab.psu.edu/publicData/) |

Copies of the dataset are included in the `Data/` directory.

---

## Methodology

### Feature Selection

From the 93 available measurements, **14 torso and thigh measurements** were selected as the most relevant for garment sizing:

| # | Measurement |
|---|---|
| 1 | Bicristal breadth |
| 2 | Buttock circumference |
| 3 | Buttock depth |
| 4 | Chest breadth |
| 5 | Chest circumference |
| 6 | Chest depth |
| 7 | Hip breadth |
| 8 | Lower thigh circumference |
| 9 | Shoulder circumference |
| 10 | Thigh circumference |
| 11 | Vertical trunk circumference (USA) |
| 12 | Waist breadth |
| 13 | Waist circumference |
| 14 | Waist depth |

### Data Preprocessing

- **Missing value analysis**: visualised using a heatmap; any incomplete records are handled prior to clustering.
- **Standardisation (Z-score normalisation)**: each feature is centred and scaled to unit variance to ensure all measurements contribute equally during clustering.

### Clustering Algorithms

Two unsupervised clustering methods are compared:

| Method | Description |
|---|---|
| **K-Medoids (PAM)** | Partitions data around actual data points (medoids), making it robust to outliers. The optimal number of clusters is determined via the Elbow Method. |
| **Agglomerative Hierarchical Clustering (Ward's linkage)** | Builds a bottom-up dendrogram by minimising within-cluster variance at each merge step. |

### Cluster Evaluation

- **Elbow Method**: plots distortion or inertia against the number of clusters *k* to identify the optimal *k* where gains diminish.
- **Davies-Bouldin Index (DBI)**: measures the ratio of within-cluster scatter to between-cluster separation; lower values indicate better-separated, compact clusters.
- **PCA Projection**: reduces the 14-dimensional space to 2 principal components for visual inspection of cluster separation.
- **Dendrogram**: hierarchical structure visualisation used to determine the appropriate cut level for Ward's method.

### Morphotype Extraction

For each identified cluster, a **representative morphotype** is extracted as the **medoid** — the data point closest to the cluster centroid by Euclidean or squared Euclidean distance. This provides a real, interpretable body shape that is most representative of all members in that cluster.

---

## Results

| Dataset | Method | Optimal *k* |
|---|---|---|
| Female | K-Medoids (PAM) | 6 |
| Female | Hierarchical (Ward) | 4 |
| Male | K-Medoids (PAM) | 6 |
| Male | Hierarchical (Ward) | 4–5 |

Key visualisations (available in `Images/`):

| File | Description |
|---|---|
| `FMedoidsProjection.png` | PCA projection of female K-Medoids clusters |
| `FHierProjection.png` | PCA projection of female hierarchical clusters |
| `MMedoidsProjection.png` | PCA projection of male K-Medoids clusters |
| `MHierProjection.png` | PCA projection of male hierarchical clusters |
| `FDendrogram.png` | Female body shape dendrogram |
| `MDendrogram.png` | Male body shape dendrogram |
| `FMedoidsElbow.png` | Elbow curve (female, K-Medoids) |
| `FHierElbow.png` | Elbow curve (female, hierarchical) |
| `FMedoidsDBI.png` | Davies-Bouldin index (female, K-Medoids) |
| `FHierDBI.png` | Davies-Bouldin index (female, hierarchical) |

Full analysis details are available in [`report.pdf`](report.pdf) and [`slide.pdf`](slide.pdf).

---

## Repository Structure

```
Projet-DiTex-Morphotypes/
│
├── Data/                          # ANSUR II dataset (female & male CSV files)
│   ├── ANSUR_II_FEMALE_Public.csv
│   └── ANSUR_II_MALE_Public.csv
│
├── Images/                        # All figures and visualisations
│
├── Morphotypes/                   # Morphotype extraction outputs and illustrations
│
├── notebook/                      # Jupyter notebooks
│   ├── Females.ipynb              # Full analysis pipeline for female data
│   └── Males.ipynb                # Full analysis pipeline for male data
│
├── docs/                          # Background literature (reference PDFs)
├── tex/                           # LaTeX source of the project report
├── V0/                            # Early version slides and report (V0)
│
├── report.pdf                     # Final project report
├── slide.pdf                      # Final presentation slides
├── Human_Body_Shapes_Anomaly_Detection_and_Classifica-1.pdf
│                                  # Related anomaly detection reference
└── README.md
```

---

## Getting Started

### Requirements

Install the required Python packages:

```bash
pip install pandas numpy seaborn scipy scikit-learn matplotlib scikit-learn-extra yellowbrick
```

| Package | Role |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical computation |
| `seaborn` | Statistical data visualisation |
| `scipy` | Hierarchical clustering and linkage |
| `scikit-learn` | PCA, K-Means, Agglomerative Clustering, metrics |
| `matplotlib` | Plotting |
| `scikit-learn-extra` | K-Medoids (PAM) implementation |
| `yellowbrick` | Elbow Method visualiser |

### Running the Notebooks

**Option 1 — Google Colaboratory (recommended, no local setup needed):**

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `notebook/Females.ipynb` or `notebook/Males.ipynb`.
3. Run all cells.

**Option 2 — Local Jupyter:**

```bash
git clone https://github.com/StevedeRose/Projet-DiTex-Morphotypes.git
cd Projet-DiTex-Morphotypes
pip install pandas numpy seaborn scipy scikit-learn matplotlib scikit-learn-extra yellowbrick
jupyter notebook notebook/Females.ipynb
```

---

## Step-by-Step Code Walkthrough

### 1. Import packages

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

### 2. Load the ANSUR II database

```py
# Female dataset
df = pd.read_csv('http://tools.openlab.psu.edu/publicData/ANSUR_II_FEMALE_Public.csv')

# Male dataset
df = pd.read_csv('http://tools.openlab.psu.edu/publicData/ANSUR_II_MALE_Public.csv', encoding='ISO-8859-1')
```

Alternatively, load from local copies in `Data/`:

```py
df = pd.read_csv('../Data/ANSUR_II_FEMALE_Public.csv')
```

### 3. Exploratory data analysis

```py
# Missing value heatmap
plt.figure(figsize=(15, 9))
sns.heatmap(df.isnull(), cmap='viridis')
plt.show()

# Full correlation heatmap
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(df.corr(), cmap='jet')
plt.show()
```

### 4. Feature extraction (14 measurements)

```py
columns = [
    'bicristalbreadth', 'buttockcircumference', 'buttockdepth',
    'chestbreadth', 'chestcircumference', 'chestdepth',
    'hipbreadth', 'lowerthighcircumference', 'shouldercircumference',
    'thighcircumference', 'verticaltrunkcircumferenceusa',
    'waistbreadth', 'waistcircumference', 'waistdepth'
]
df_fit_ = pd.DataFrame(df, columns=columns)

# Correlation matrix of selected features
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(df_fit_.corr(), cmap='jet')
plt.show()
```

### 5. Standardisation

```py
df_fit = (df_fit_.copy() - df_fit_.mean()) / df_fit_.std()
```

### 6. Determine optimal number of clusters (Elbow Method)

```py
model = KMedoids(method='pam')  # or AgglomerativeClustering(linkage='ward')
visualizer = KElbowVisualizer(model, k=(2, 16), timings=False)
visualizer.fit(df_fit)
visualizer.show()
```

### 7. Cluster and visualise (PCA projection)

```py
colors = ['ro', 'g^', 'bs', 'cX', 'yP', 'm*', 'kH']

def clustering(data, method, plot=True):
    method.fit(data)
    identified_clusters = method.fit_predict(data)

    acp = PCA(n_components=2, svd_solver='full')
    acp.fit_transform(data)

    if not plot:
        print(100 * sum(acp.explained_variance_ratio_), '% of the variance retained')
    else:
        x_proj = acp.transform(data)
        x_proj = pd.DataFrame(data=x_proj,
                               columns=['Composante principale 1', 'Composante principale 2'])
        x_proj['Clusters'] = identified_clusters

        fig, ax = plt.subplots(figsize=(25, 25))
        plt.xlabel('First component: {:.2f}%'.format(100 * acp.explained_variance_ratio_[0]), fontsize=20)
        plt.ylabel('Second component: {:.2f}%'.format(100 * acp.explained_variance_ratio_[1]), fontsize=20)
        for i in range(x_proj['Clusters'].max() + 1):
            plt.plot(x_proj[x_proj['Clusters'] == i]['Composante principale 1'],
                     x_proj[x_proj['Clusters'] == i]['Composante principale 2'],
                     colors[i], label='Cluster {}'.format(i + 1), alpha=0.7)
        plt.legend(loc='best', fontsize=20)
        plt.show()

# K-Medoids (6 clusters)
clustering(df_fit, KMedoids(n_clusters=6, method='pam'))

# Ward's hierarchical clustering (4 clusters)
clustering(df_fit, AgglomerativeClustering(n_clusters=4, linkage='ward'))
```

### 8. Dendrogram (hierarchical clustering)

```py
plt.figure(figsize=(12, 6))
plt.title("Body Shape Dendrogram")
dend = shc.dendrogram(shc.linkage(df_fit, method='ward'),
                      p=50, truncate_mode='lastp', show_leaf_counts=False)
plt.tight_layout()
plt.show()
```

### 9. Cluster description and morphotype extraction

```py
# Fit and assign clusters
method = KMedoids(n_clusters=6, method='pam')
method.fit(df_fit)
identified_clusters = method.fit_predict(df_fit)

df_means = df.select_dtypes(include='number').copy()
df_means["Cluster"] = identified_clusters.copy()
df_means["Weight"] = df_means["Weightlbs"].copy() * 0.453592   # lbs → kg
df_means["Height"] = df_means["Heightin"].copy() * 2.54        # in → cm
df_means["BMI"] = 10000 * df_means['Weight'] / df_means['Height'] ** 2
df_fit_["Cluster"] = identified_clusters.copy()

# Drop irrelevant columns
for col in ['subjectid', 'SubjectNumericRace', 'DODRace', 'Weightlbs', 'weightkg', 'Heightin']:
    df_means.pop(col)

# Describe a specific cluster (e.g., cluster 2)
cluster2 = df_means[df_means["Cluster"] == 1].copy()
cluster2.pop('Cluster')
cluster2.describe()
```

#### Find the medoid (most representative individual) of cluster *n*

```py
df_medoids = df_fit.copy()
df_medoids["Cluster"] = identified_clusters.copy()

# Euclidean distance
df_medoid_n = df_medoids[df_medoids["Cluster"] == n - 1]
medoid_n = np.argmin(np.sqrt(((df_medoid_n - df_medoid_n.mean()) ** 2).sum(axis=1)))

# Full measurements of the medoid
df_fit_[df_fit_["Cluster"] == n - 1].iloc[medoid_n]
```

---

## References

The `docs/` directory contains the full bibliography used in this project. Key references include:

- Simmons, K., Istook, C., & Devarajan, P. (2004). *Female Figure Identification Technique (FFIT) for Apparel* — Parts I & II.
- Hamad, M., Thomassey, S., & Bruniaux, P. (2017). *A new sizing system based on 3D shape descriptor for morphology clustering*.
- Nakamura, M., & Kurokawa, T. (2009). *Analysis and classification of three-dimensional trunk shape of women using the human body shape model*.
- Cottle, D. (2012). *Statistical Human Body Form Classification Methodology Development*.
- Park, J., & Park, S. (2013). *Body shape analyses of large persons in South Korea*.

---

## Authors

This project was carried out by Master's students at the **University of Technology of Troyes (UTT)** in collaboration with **DiTeX** (2021).

| Institution | Logo |
|---|---|
| University of Technology of Troyes | ![UTT](Images/Logo_UTT.png) |
