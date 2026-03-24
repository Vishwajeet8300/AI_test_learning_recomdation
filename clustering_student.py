import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score

# Read the CSV file from your local directory
df = pd.read_csv(r"C:\Users\Shardul More\OneDrive\Desktop\project-root\newpro\student_weak_subjects.csv")  # Replace with your file path

# Selecting the columns we will use for clustering
numeric_columns = ['Machine Learning', 'Computer Networks']
numeric_df = df[numeric_columns]

# Handle missing values by imputing
imputer = SimpleImputer(strategy='mean')
numeric_df = pd.DataFrame(imputer.fit_transform(numeric_df), columns=numeric_df.columns)

# Perform Agglomerative Clustering
model = AgglomerativeClustering(n_clusters=4)
df['Cluster'] = model.fit_predict(numeric_df)

# Visualize Clusters
plt.figure(figsize=(8, 5))
plt.scatter(df['Machine Learning'], df['Computer Networks'], c=df['Cluster'], cmap='rainbow', s=50)
centroids = df.groupby('Cluster')[['Machine Learning', 'Computer Networks']].mean()
plt.scatter(centroids['Machine Learning'], centroids['Computer Networks'], c='black', marker='x', s=100, label='Centroids')
plt.xlabel('Machine Learning')
plt.ylabel('Computer Networks')
plt.title('Clustering of Students by Machine Learning and Computer Networks Scores')
plt.legend()
plt.show()

# Save the clustered data to CSV
df.to_csv('clustered_students_no_norm.csv', index=False)

# Show clustered DataFrame
df.head()
