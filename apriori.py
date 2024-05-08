# Import necessary libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Load the Groceries dataset
df = pd.read_csv("http://www.amlbook.com/data/zip/groceries.zip", compression="zip", header=None, sep=",")

# Convert the dataset into transaction format
transactions = []
for i in range(0, len(df)):
    transactions.append([str(df.values[i, j]) for j in range(0, len(df.columns))])

# Convert the transactions to a dataframe
df = pd.DataFrame(transactions)

# Generate one-hot encoded format
oht = df.stack().str.get_dummies().sum(level=0)

# Apply the Apriori algorithm
frequent_itemsets = apriori(oht, min_support=0.01, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Print frequent itemsets
print("Frequent Itemsets:")
print(frequent_itemsets)

# Print association rules
print("\nAssociation Rules:")
print(rules)
