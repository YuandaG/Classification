import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from saxpy.sax import sax

# Step 1: Data Preprocessing
def preprocess_data(load_data):
    # Remove outliers (3σ rule)
    mean = np.mean(load_data, axis=0)
    std_dev = np.std(load_data, axis=0)
    filtered_data = load_data[np.abs(load_data - mean) < 3 * std_dev]
    
    # Normalize using Z-score
    normalized_data = zscore(filtered_data, axis=0)
    return normalized_data

# Step 2: Dimension Reduction using PAA
def apply_paa(data, num_segments):
    segment_size = len(data) // num_segments
    paa_representation = [np.mean(data[i * segment_size:(i + 1) * segment_size])
                          for i in range(num_segments)]
    return np.array(paa_representation)

# Step 3: Symbolic Representation using SAX
def apply_sax(data, num_segments, alphabet_size):
    paa_representation = apply_paa(data, num_segments)
    sax_representation = sax.to_letter_rep(paa_representation, alphabet_size)
    return sax_representation

# Step 4: Clustering using K-means
def cluster_profiles(data, max_clusters=10):
    distortions = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        distortions.append(kmeans.inertia_)
    
    # Plot the elbow curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_clusters + 1), distortions, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion')
    plt.show()
    
    # Choose the optimal number of clusters (user input or elbow observation)
    optimal_clusters = int(input("Enter the optimal number of clusters: "))
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=0).fit(data)
    return kmeans.labels_

# Example Usage
if __name__ == "__main__":
    # Load example data (replace with real load data)
    num_samples = 365  # Example: Daily load data for one year
    num_points = 96    # Example: 96 sampling points per day
    example_data = np.random.rand(num_samples, num_points)

    # Preprocess
    preprocessed_data = preprocess_data(example_data)

    # Apply PAA and SAX for each profile
    num_segments = 8
    alphabet_size = 4
    sax_profiles = np.array([apply_sax(profile, num_segments, alphabet_size) for profile in preprocessed_data])

    # Convert SAX words to numeric representation for clustering
    numeric_profiles = np.array([[ord(char) for char in profile] for profile in sax_profiles])

    # Cluster and visualize
    cluster_labels = cluster_profiles(numeric_profiles)
    print("Cluster Labels:", cluster_labels)