import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the Dataset
df = pd.read_csv("c:/Users/HP/Desktop/task2/Mall_Customers.csv")

# Display the first few rows of the dataframe
print(df.head())

# Step 2: Data Preprocessing
# Select the relevant features
features = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 3: K-means Clustering
# Choosing the number of clusters (k) using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i, init="k-means++", max_iter=300, n_init=10, random_state=42
    )
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Method graph
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

# Choose the optimal number of clusters (k)
k = 5
kmeans = KMeans(
    n_clusters=k, init="k-means++", max_iter=300, n_init=10, random_state=42
)
clusters = kmeans.fit_predict(scaled_features)

# Add cluster labels to the original dataframe
df["Cluster"] = clusters

# Step 4: Evaluation
# Calculate silhouette score
score = silhouette_score(scaled_features, clusters)
print(f"Silhouette Score: {score}")

# Optional: Visualize the clustering (2D plot)
plt.figure(figsize=(10, 5))
plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"],
    cmap="viridis",
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation")
plt.colorbar()
plt.show()

# Save the clustered data
df.to_csv("customer_clusters.csv", index=False)

print(df.head())

# Step 5: Plotting Distribution Plots
plt.figure(figsize=(15, 5))

# Distribution plot for Age
plt.subplot(1, 3, 1)
sns.histplot(df["Age"], kde=True)
plt.title("Age Distribution")

# Distribution plot for Annual Income
plt.subplot(1, 3, 2)
sns.histplot(df["Annual Income (k$)"], kde=True)
plt.title("Annual Income Distribution")

# Distribution plot for Spending Score
plt.subplot(1, 3, 3)
sns.histplot(df["Spending Score (1-100)"], kde=True)
plt.title("Spending Score Distribution")

plt.tight_layout()
plt.show()
