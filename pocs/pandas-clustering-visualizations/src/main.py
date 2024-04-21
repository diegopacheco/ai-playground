import pandas as pd
from faker import Faker
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Initialize Faker
fake = Faker()

# Define the number of records
num_records = 1000

# Define the list of products
products = ['Minecraft', 'Fortnite', 'Among Us', 'Call of Duty', 'Cyberpunk 2077']

# Generate the data
data = {
    'Customer Name': [fake.name() for _ in range(num_records)],
    'Transaction ID': [fake.unique.random_number(digits=5) for _ in range(num_records)],
    'Product': [random.choice(products) for _ in range(num_records)],
    'Quantity': [random.randint(1, 10) for _ in range(num_records)],
    'Transaction Date': [fake.date_between(start_date='-1y', end_date='today') for _ in range(num_records)]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Print the DataFrame
print(df)

def display_data_with_kmeans(df, n_clusters):
    # Convert 'Product' column to numerical values
    le = LabelEncoder()
    df['Product'] = le.fit_transform(df['Product'])

    # Define the features to be used in KMeans
    features = ['Product', 'Quantity']

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(df[features])

    # Add the cluster labels to the DataFrame
    df['Cluster'] = kmeans.labels_

    # Plot the data
    plt.scatter(df['Product'], df['Quantity'], c=df['Cluster'])
    plt.xlabel('Product')
    plt.ylabel('Quantity')
    plt.title('Customer Transactions - KMeans Clustering')
    plt.show()

def display_data_with_gmm(df, n_components):
    # Convert 'Product' column to numerical values
    le = LabelEncoder()
    df['Product'] = le.fit_transform(df['Product'])

    # Define the features to be used in GMM
    features = ['Product', 'Quantity']

    # Perform GMM
    gmm = GaussianMixture(n_components=n_components, random_state=0).fit(df[features])

    # Predict the cluster for each data point
    labels = gmm.predict(df[features])

    # Add the cluster labels to the DataFrame
    df['Cluster'] = labels

    # Plot the data
    plt.scatter(df['Product'], df['Quantity'], c=df['Cluster'])
    plt.xlabel('Product')
    plt.ylabel('Quantity')
    plt.title('Customer Transactions - Gaussian Mixture Model')
    plt.show()

def display_data_with_dbscan(df, eps, min_samples):
    # Convert 'Product' column to numerical values
    le = LabelEncoder()
    df['Product'] = le.fit_transform(df['Product'])

    # Define the features to be used in DBSCAN
    features = ['Product', 'Quantity']

    # Perform DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(df[features])

    # Add the cluster labels to the DataFrame
    df['Cluster'] = dbscan.labels_

    # Plot the data
    plt.scatter(df['Product'], df['Quantity'], c=df['Cluster'])
    plt.xlabel('Product')
    plt.ylabel('Quantity')
    plt.title('Customer Transactions - DBSCAN')
    plt.show()

display_data_with_kmeans(df, 3)
display_data_with_gmm(df, 3)
display_data_with_dbscan(df, 0.5, 5)