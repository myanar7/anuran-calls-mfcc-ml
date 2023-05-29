from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import pandas as pd

# Load dataset
data = data = pd.read_csv('dataset/Frogs_MFCCs.csv')

print("###################################### FREQUENT PATTERN MINING MODELS ###########################################")

# Get MFCC features
features = data.columns[-4:]  # Let us assume that the last 4 columns are the MFCC features

# Transform data into a list of lists
transactions = data[features].astype(str).values.tolist()

# Encode transactions
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Implement Apriori algorithm
frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)

# Extract association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Print results
print("Apriori Frequent Itemsets:")
print(frequent_itemsets)
print("\nApriori Association Rules:")
print(rules)

# Implement FP-Growth algorithm
frequent_itemsets_fp = fpgrowth(df_encoded, min_support=0.1, use_colnames=True)

# Extract association rules
rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=0.5)

# Print results
print("FP-Growth Frequent Itemsets:")
print(frequent_itemsets_fp)
print("\nFP-Growth Association Rules:")
print(rules_fp)

# Implement ECLAT algorithm
frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)

# Extract association rules
rules = association_rules(frequent_itemsets,metric='support',min_threshold= 0.1)

# Print results
print("ECLAT Frequent Itemsets:")
print(frequent_itemsets)
print("\nECLAT Association Rules:")
print(rules)

print("###################################### CLUSTERING ANALYSIS METHODS ###########################################")

df = pd.DataFrame(data, columns=['MFCCs_ 1', 'MFCCs_ 2', 'MFCCs_ 3', 'MFCCs_ 4', 'MFCCs_ 5', 'MFCCs_ 6', 'MFCCs_ 7', 'MFCCs_ 8', 'MFCCs_ 9', 'MFCCs_10', 'MFCCs_11', 'MFCCs_12', 'MFCCs_13', 'MFCCs_14', 'MFCCs_15', 'MFCCs_16', 'MFCCs_17', 'MFCCs_18', 'MFCCs_19', 'MFCCs_20', 'MFCCs_21', 'MFCCs_22', 'Family', 'Genus', 'Species', 'RecordID'])
# Implement K-Means algorithm
k = 4  # Count of clusters
features = df[['MFCCs_ 1', 'MFCCs_ 2', 'MFCCs_ 3', 'MFCCs_ 4', 'MFCCs_ 5', 'MFCCs_ 6', 'MFCCs_ 7', 'MFCCs_ 8', 'MFCCs_ 9', 'MFCCs_10', 'MFCCs_11', 'MFCCs_12', 'MFCCs_13', 'MFCCs_14', 'MFCCs_15', 'MFCCs_16', 'MFCCs_17', 'MFCCs_18', 'MFCCs_19', 'MFCCs_20', 'MFCCs_21', 'MFCCs_22']]
kmeans = KMeans(n_clusters=k, random_state=42, n_init=4)
kmeans.fit(features)

# Get cluster labels
labels = kmeans.labels_

# Add cluster labels to the dataset
df['Cluster'] = labels

# Calculate WCSS
wcss = kmeans.inertia_

# Calculate Silhouette Score
silhouette_avg = silhouette_score(features, labels)

# Print results
print(df.sample(5))

print("WCSS:", wcss)
print("Silhouette Coefficient:", silhouette_avg)

# Implement Agglomerative Clustering algorithm
agg_clustering = AgglomerativeClustering(n_clusters=4)  # Set the number of clusters
clusters = agg_clustering.fit_predict(features)

# Add cluster labels to the dataset
df['Cluster'] = clusters

# Calculate Silhouette Score
print(df.sample(5))

silhouette_avg = silhouette_score(features, clusters)
print("Silhouette Score:", silhouette_avg)

# Implement DBSCAN algorithm
dbscan = DBSCAN(eps=1.1, min_samples=5)  # Set eps and min_samples
clusters = dbscan.fit_predict(features)

# Show the results
print(df.sample(5))

silhouette_avg = silhouette_score(features, clusters)
print("Silhouette Score:", silhouette_avg)

