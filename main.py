from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
import pandas as pd

# Veri setini yükleyin
data = data = pd.read_csv('dataset/Frogs_MFCCs.csv')

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