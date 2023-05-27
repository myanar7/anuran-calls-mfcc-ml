from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import pandas as pd

# Veri setini yükleyin
data = data = pd.read_csv('dataset/Frogs_MFCCs.csv')

print("###################################### FREQUENT PATTERN MINING MODELS ###########################################")

# MFCC özelliklerini alın
features = data.columns[-4:]  # Son dört sütun Family, Genus, Species ve RecordID olduğunu varsayalım

# Verileri uygun formata dönüştürün
transactions = data[features].astype(str).values.tolist()

# Verileri TransactionEncoder ile kodlayın
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Apriori algoritmasını uygulayın
frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)

# İlişkisel kuralları çıkarın
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Sonuçları görüntüle
print("Sık Öğeler:")
print(frequent_itemsets)
print("\nİlişkisel Kurallar:")
print(rules)

# FP-Growth algoritmasını uygulayın
frequent_itemsets_fp = fpgrowth(df_encoded, min_support=0.1, use_colnames=True)

# İlişkisel kuralları çıkarın
rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=0.5)

# Sonuçları görüntüle
print("FP-Growth ile Sık Öğeler:")
print(frequent_itemsets_fp)
print("\nFP-Growth ile İlişkisel Kurallar:")
print(rules_fp)

# ECLAT algoritmasını uygulayın
frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)

# İlişkisel kuralları çıkarın
rules = association_rules(frequent_itemsets,metric='support',min_threshold= 0.1)

# Sonuçları görüntüle
print("Sık Öğeler:")
print(frequent_itemsets)
print("\nİlişkisel Kurallar:")
print(rules)

print("###################################### CLUSTERING ANALYSIS METHODS ###########################################")

#print(data.describe())

df = pd.DataFrame(data, columns=['MFCCs_ 1', 'MFCCs_ 2', 'MFCCs_ 3', 'MFCCs_ 4', 'MFCCs_ 5', 'MFCCs_ 6', 'MFCCs_ 7', 'MFCCs_ 8', 'MFCCs_ 9', 'MFCCs_10', 'MFCCs_11', 'MFCCs_12', 'MFCCs_13', 'MFCCs_14', 'MFCCs_15', 'MFCCs_16', 'MFCCs_17', 'MFCCs_18', 'MFCCs_19', 'MFCCs_20', 'MFCCs_21', 'MFCCs_22', 'Family', 'Genus', 'Species', 'RecordID'])
# K-means modelini oluştur ve uygula
k = 4  # Küme sayısı
features = df[['MFCCs_ 1', 'MFCCs_ 2', 'MFCCs_ 3', 'MFCCs_ 4', 'MFCCs_ 5', 'MFCCs_ 6', 'MFCCs_ 7', 'MFCCs_ 8', 'MFCCs_ 9', 'MFCCs_10', 'MFCCs_11', 'MFCCs_12', 'MFCCs_13', 'MFCCs_14', 'MFCCs_15', 'MFCCs_16', 'MFCCs_17', 'MFCCs_18', 'MFCCs_19', 'MFCCs_20', 'MFCCs_21', 'MFCCs_22']]
kmeans = KMeans(n_clusters=k, random_state=42, n_init=4)
kmeans.fit(features)

# Küme etiketlerini al
labels = kmeans.labels_

# Kümeleme sonuçlarını DataFrame'e ekle
df['Cluster'] = labels

# WCSS metriğini hesapla
wcss = kmeans.inertia_

# Silhouette Coefficient metriğini hesapla
silhouette_avg = silhouette_score(features, labels)

# Sonuçları görüntüle
print(df.sample(5))

print("WCSS:", wcss)
print("Silhouette Coefficient:", silhouette_avg)

# AgglomerativeClustering modelini oluşturma ve uygulama
agg_clustering = AgglomerativeClustering(n_clusters=4)  # Küme sayısını isteğe bağlı olarak belirleyin
clusters = agg_clustering.fit_predict(features)

# Sonuçları elde etme
df['Cluster'] = clusters

# Sonuçları görüntüleme
print(df.sample(5))

silhouette_avg = silhouette_score(features, clusters)
print("Silhouette Score:", silhouette_avg)

# DBSCAN modelini oluşturma ve uygulama
dbscan = DBSCAN(eps=1.1, min_samples=5)  # eps ve min_samples değerlerini isteğe bağlı olarak ayarlayın
clusters = dbscan.fit_predict(features)

# Sonucu görüntüleme
print(df.sample(5))

silhouette_avg = silhouette_score(features, clusters)
print("Silhouette Skoru:", silhouette_avg)

