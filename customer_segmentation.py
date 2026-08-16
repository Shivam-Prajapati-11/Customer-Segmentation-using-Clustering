#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Customer Segmentation using RFM Analysis and K-Means.
Segments an online retailer's customers using RFM features
(Recency, Frequency, Monetary) and K-Means clustering.
Run with:  python customer_segmentation.py
"""

# # Customer Segmentation using RFM Analysis and K-Means
# A beginner-friendly project that groups customers of an online retailer using
# **RFM** (Recency, Frequency, Monetary) features and **K-Means clustering**, then
# turns each group into a persona with a marketing action.
#
# Dataset: UCI / Kaggle **Online Retail** (e-commerce transactions from a UK retailer).


# ## 1. Import Libraries


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

from pathlib import Path
import kagglehub

# plotting style
sns.set_style("whitegrid")
print("Libraries imported")


# ## 2. Load Dataset
# Download the Online Retail dataset from Kaggle and read it into a DataFrame.


# Download the dataset (this may take a minute on first run)
data_dir = kagglehub.dataset_download("carrie1/ecommerce-data")
csv_file = list(Path(data_dir).rglob("data.csv"))[0]

# Read the CSV with an encoding fallback
for enc in ("utf-8", "latin-1", "cp1252"):
    try:
        df = pd.read_csv(csv_file, encoding=enc)
        print(f"Loaded with encoding: {enc}")
        break
    except UnicodeDecodeError:
        continue

print("Shape:", df.shape)
print(df.head())


# ## 3. Data Cleaning
# Drop rows with a missing CustomerID, cancelled orders, and negative/zero
# quantity or price so the data only contains real purchases.


# Parse the invoice date
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%m/%d/%Y %H:%M")

# Drop missing CustomerID
df_clean = df[df["CustomerID"].notna()].copy()

# Drop cancelled orders (InvoiceNo starts with "C")
df_clean = df_clean[~df_clean["InvoiceNo"].str.startswith("C")]

# Drop rows with negative or zero quantity / price
df_clean = df_clean[(df_clean["Quantity"] > 0) & (df_clean["UnitPrice"] > 0)]

# Convert CustomerID to int (no more missing values to worry about)
df_clean["CustomerID"] = df_clean["CustomerID"].astype(int)

print("Before cleaning:", df.shape)
print("After cleaning:", df_clean.shape)
print("\nMissing values after cleaning:")
print(df_clean.isnull().sum())
print(df_clean.head())


# ## 4. Feature Engineering (RFM)
# Create a **TotalPrice** column, then compute **Recency**, **Frequency** and
# **Monetary** per customer and combine them into one `rfm` dataframe.


# Money spent per row
df_clean["TotalPrice"] = df_clean["Quantity"] * df_clean["UnitPrice"]

# Reference date = one day after the last purchase
reference_date = df_clean["InvoiceDate"].max() + pd.Timedelta(days=1)

# Group by customer and build the RFM features
rfm = df_clean.groupby("CustomerID").agg({
    "InvoiceDate": "max",       # last purchase date  -> Recency
    "InvoiceNo": "nunique",     # number of purchases -> Frequency
    "TotalPrice": "sum"         # total money spent   -> Monetary
})
rfm.columns = ["Recency", "Frequency", "Monetary"]

# Recency = days since the customer last bought something
rfm["Recency"] = (reference_date - rfm["Recency"]).dt.days

print("Reference date:", reference_date)
print("Number of customers:", rfm.shape[0])
print(rfm.head().sort_values("Monetary", ascending=False))


# ## 5. Exploratory Data Analysis
# Look at the distribution of each RFM feature and spot outliers with boxplots.


fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, rfm.columns):
    sns.histplot(rfm[col], kde=True, ax=ax)
    ax.set_title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

print(rfm.describe())


fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, rfm.columns):
    sns.boxplot(x=rfm[col], ax=ax)
    ax.set_title(f"Boxplot of {col}")
plt.tight_layout()
plt.show()


# Check how skewed the features are
print("Skewness:")
print(rfm.skew())


# ## 6. Preprocessing
# The features are heavily right-skewed, so cap the extremes at the 99th
# percentile, then standardize everything so the features are comparable.


# Cap outliers at the 99th percentile
freq_cap = rfm["Frequency"].quantile(0.99)
money_cap = rfm["Monetary"].quantile(0.99)

rfm_capped = rfm.copy()
rfm_capped["Frequency"] = rfm_capped["Frequency"].clip(upper=freq_cap)
rfm_capped["Monetary"] = rfm_capped["Monetary"].clip(upper=money_cap)

print(f"Frequency cap: {freq_cap}")
print(f"Monetary cap: {money_cap}")

# Standardize the features (mean ~ 0, std ~ 1)
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_capped[["Recency", "Frequency", "Monetary"]])
rfm_scaled = pd.DataFrame(rfm_scaled, columns=rfm.columns, index=rfm.index)

print(rfm_scaled.describe().round(2))


# ## 7. Finding Optimal K
# Use the **elbow method** (WCSS) and **silhouette score** to choose how many
# clusters to use.


# Inertia / WCSS for the elbow method
wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, random_state=42, n_init=10)
    km.fit(rfm_scaled)
    wcss.append(km.inertia_)

# Silhouette scores (needs at least 2 clusters)
sil_scores = []
for i in range(2, 11):
    km = KMeans(n_clusters=i, random_state=42, n_init=10)
    labels = km.fit_predict(rfm_scaled)
    sil_scores.append(silhouette_score(rfm_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(range(1, 11), wcss, "o-")
axes[0].set_title("Elbow Method (WCSS)")
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("WCSS")

axes[1].plot(range(2, 11), sil_scores, "o-")
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Silhouette Score")
plt.tight_layout()
plt.show()

print("WCSS values:", wcss)
print("Silhouette scores:", sil_scores)
print("\nBased on the plots, k = 4 looks like a good choice")

# chosen number of clusters
k = 4


# ## 8. K-Means Clustering
# Fit K-Means with the chosen `k` and add the cluster labels to `rfm`.


# Fit the model on the standardized features
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(rfm_scaled)

# Add the cluster number to each customer
rfm["Cluster"] = kmeans.labels_
print(rfm.head())


# ## 9. Cluster Stability Check
# Re-run K-Means with a few different `random_state` values and compare the
# cluster assignments with the Adjusted Rand Index (high = stable clusters).


ari_scores = []
base_labels = kmeans.labels_

for random_state in [0, 10, 42, 100]:
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    km.fit(rfm_scaled)
    ari = adjusted_rand_score(base_labels, km.labels_)
    ari_scores.append(ari)

print("ARI scores:", ari_scores)
print("Average stability score:", round(np.mean(ari_scores), 3))


# ## 10. Cluster Profiling
# See what each cluster looks like on average, rank them by value, give each a
# persona name, and plot the clusters.


# Average RFM and size per cluster
cluster_means = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean().round(2)
print("Average RFM per cluster:")
print(cluster_means)
print("\nCustomers per cluster:")
print(rfm["Cluster"].value_counts().sort_index())

# Order clusters and name them: value decides the top two, inactivity decides the rest
rank = cluster_means["Monetary"].sort_values(ascending=False).index
personas = {rank[0]: "Champions", rank[1]: "Loyal"}

# Of the two lower-value clusters, the one inactive the longest is At-Risk
low_value = list(rank[2:])
inactive = cluster_means.loc[low_value, "Recency"].sort_values(ascending=False).index
personas[inactive[0]] = "At-Risk"
personas[inactive[1]] = "New/Low-Value"

rfm["Persona"] = rfm["Cluster"].map(personas)
print("\nPersona map:", personas)

# Scatter plot of the scaled features colored by cluster
plot_data = rfm_scaled.copy()
plot_data["Cluster"] = rfm["Cluster"]
sns.pairplot(plot_data, vars=["Recency", "Frequency", "Monetary"], hue="Cluster")
plt.show()


# ## 11. Segment-to-Action Mapping
# Turn each persona into a clear marketing action and build a summary table.


# Marketing action for each persona
actions = {
    "Champions":  "Reward with exclusive offers and loyalty perks",
    "Loyal":       "Send repeat-purchase offers to keep them engaged",
    "At-Risk":     "Reactivation campaign with a welcome-back discount",
    "New/Low-Value": "Affordable deals and intro offers to grow them"
}

# Build the final summary table
summary = cluster_means.reset_index()
summary["persona"] = summary["Cluster"].map(personas)
summary["customer_count"] = summary["Cluster"].map(rfm["Cluster"].value_counts().sort_index())
summary["avg_monetary"] = summary["Monetary"].round(2)
summary["avg_recency"] = summary["Recency"].round(1)
summary["recommended_action"] = summary["persona"].map(actions)

summary = summary[["Cluster", "persona", "customer_count",
                   "avg_recency", "avg_monetary", "recommended_action"]]
print(summary)


# ## 12. Business Insights
# **What these segments mean for the business:**
#
# - **Champions** are your most valuable customers — they buy often, spend the most,
#   and have bought recently. Reward them with exclusive perks and VIP treatment to
#   keep them loyal.
#
# - **Loyal** customers spend well and shop regularly. Keep them engaged with
#    repeat-purchase offers, bundles, and a solid loyalty program to push them up
#    into Champion territory.
#
# - **At-Risk** customers used to be valuable but have not shopped in a while.
#    A targeted reactivation campaign (discounts or 'we miss you' emails) is the
#    cheapest way to win back lost revenue.
#
# - **New/Low-Value** customers are recent or low-spenders. Nurturing them with
#    onboarding emails and first-purchase discounts helps build habits and move
#    them towards higher-value segments.
#
# - Clustering on RFM keeps each segment actionable — each persona maps to a
#    specific marketing action, so the analysis translates directly into decisions.

