
#  Machine Learning Fundamentals — Day-by-Day Practice

A full **20-day self-learning journey** covering **Supervised** and **Unsupervised Learning** techniques — from simple regression models to dimensionality reduction and clustering evaluation.

This repository summarizes the **entire practical foundation of Machine Learning**, including data preprocessing, model building, evaluation, and visualization.

---

##  Overview

| Day | Topic | Description |
|-----|--------|-------------|
| 1 | Simple Linear Regression | Predicting house price from size |
| 2 | Multiple Linear Regression | Predicting rent using area + rooms |
| 3 | Polynomial Regression | Curve fitting (salary vs experience²) |
| 4 | Logistic Regression (1 feature) | Study hours → pass/fail |
| 5 | Logistic Regression (2 features) | GPA + activities → accepted |
| 6 | Decision Tree Classifier | Titanic-like dataset |
| 8 | Regression Metrics | MSE, MAE, RMSE, R² |
| 9 | Classification Metrics | Confusion Matrix, Precision, Recall, F1 |
| 10 | Train/Test Split + Cross Validation | Comparing model performance consistency |
| 11 | GridSearchCV | Hyperparameter tuning (Decision Tree) |
| 13 | Gradient Boosting / XGBoost | Intro to ensemble learning |
| 14 | Mini-project | Customer churn prediction |
| 15 | KMeans Clustering | Customer segmentation |
| 16 | Hierarchical Clustering | Dendrogram & Agglomerative clustering |
| 17 | DBSCAN | Detecting outliers |
| 18 | PCA | Dimensionality reduction & variance explained |
| 19 | t-SNE | Non-linear 2D visualization (MNIST & synthetic) |
| 20 | Clustering Evaluation | Elbow method & Silhouette score |

---

## Libraries Used

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
````

---

##  Concepts Covered

###  **Supervised Learning**

* Linear, Multiple, and Polynomial Regression
* Logistic Regression (binary classification)
* Decision Trees + Hyperparameter tuning
* Model Evaluation (Train/Test Split, Cross Validation, Metrics)

###  **Unsupervised Learning**

* KMeans, Hierarchical, and DBSCAN clustering
* Dimensionality Reduction (PCA, t-SNE)
* Clustering Evaluation (Elbow, Silhouette)

###  **Visualization**

* Data relationships with `matplotlib`
* Explained variance & component analysis (PCA)
* Non-linear projections using `t-SNE`
* Cluster visualizations and dendrograms

---

##  Example Visualizations

* Linear Regression fitting line
* Decision Tree plot
* PCA explained variance ratio
* t-SNE scatter with different perplexity values
* KMeans and Hierarchical cluster plots

---

##  Next Steps

* Apply models on real-world datasets (Kaggle / UCI)
* Add evaluation reports (ROC Curve, Precision-Recall Curve)
* Implement Feature Engineering (encoding, handling missing data)
* Build an interactive ML demo using **Streamlit** or **Gradio**

---

##  Author’s Note

This project represents a complete **Machine Learning roadmap** — from basic regression to unsupervised visualization — implemented manually without using pre-made notebooks.
Perfect for **students or self-learners** who want to practice each concept independently.

