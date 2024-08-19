# PRODIGY_ML_02
Customer Segmentation with K-Means Clustering
Overview
This project uses the K-means clustering algorithm to segment customers based on their purchase history data from a retail store. The goal is to identify distinct customer groups for targeted marketing and strategic decision-making.

Dataset
The dataset used in this project is available on Kaggle. It includes the following features:

Customer ID: Unique identifier for each customer.
Gender: Gender of the customer.
Age: Age of the customer.
Annual Income (k$): Annual income of the customer in thousands of dollars.
Spending Score (1-100): Spending score assigned to the customer by the retail store.
Dataset URL: Customer Segmentation Dataset

Project Structure
data/: Contains the raw dataset and any processed files.
notebooks/: Jupyter notebooks for data exploration, preprocessing, and clustering.
src/: Source code for data processing, clustering, and evaluation.
results/: Visualizations and outputs of the clustering algorithm.
README.md: This file.
Usage
Load and Explore the Dataset:

Open notebooks/Data_Exploration.ipynb to load and explore the dataset.
Data Preprocessing:

Standardize the features using notebooks/Data_Preprocessing.ipynb.
K-means Clustering:

Determine the optimal number of clusters using the Elbow Method.
Implement K-means clustering in notebooks/KMeans_Clustering.ipynb.
Evaluation and Visualization:

Assess the clustering results with silhouette scores and visualize clusters.
Optional: Explore additional visualizations in results/.
Save Results:

The clustered data will be saved in customer_clusters.csv in the data/ directory.
Code
The core functionalities are implemented in the following scripts:

src/data_preprocessing.py: Functions for cleaning and preparing the data.
src/kmeans_clustering.py: Functions for running K-means clustering and evaluating results.
Results
Elbow Method Plot: Helps determine the optimal number of clusters.
Cluster Visualization: A scatter plot showing clusters based on Annual Income and Spending Score.
Distribution Plots: Histograms of Age, Annual Income, and Spending Score to analyze feature distributions.
Contributing
Contributions to improve the project or add new features are welcome. Please open issues or submit pull requests.
