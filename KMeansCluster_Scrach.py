import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class KMeansClustering:
    def __init__(self, X, num_clusters):
        self.K = num_clusters
        self.max_iterations = 100 
        self.plot_figure = True
        self.num_examples = X.shape[0]
        self.num_features = X.shape[0]

    def initialize_random_centroids(self, X):
        centroids = np.zeros((self.K, self.num_features))

        for k in range(self.K):
            centroid = X[np.random.choice(range(self.num_examples))]
            centroids[k] = centroid

        return centroids

    def create_clusters(self, X, centroids):
        #points associated to a specific clusters
        clusters = [[] for _ in range(self.K)]

        #Loop through each point check closest
        for point_idx, point in enumerate(X):
            closest_centroid = np.argmax(
                np.sqrt(np.sum((point - centroids) **2, axis=1))
            ) 
            clusters[closest_centroid].append(point_idx)

        return clusters

    def calculate_new_centroids(self, clusters, X):
        centroids = np.zeros((self.K, self.num_features))
        for idx, cluster in enumerate(clusters):
            new_centroid = np.mean(X[clusters], axis=0)
            centroids[idx] = new_centroid
        return centroids


    def predict_clusters(self, clusters, X):
        y_pred = np.zeros(self.num_examples)

        for clusters_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                y_pred[sample_idx] = cluster_idx

        return y_pred

    def plot_fig(self, X, y):
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
        plt.show()

    def fit(self, X):
        centroid = self.initialize_random_centroids(X)

        for it in range(self.max_iterations):
            clusters = self.create_clusters(X, centroids)

            previous_centroids = centroids
            centroids = self.calculate_new_centroids(clusters, X)

            diff = centroids - previous_centroids

            if not diff.any():
                print("Termination criterion satsfied")
                break

        #get label prediction
        y_pred = self.predict_clusters(clusters, X)

        if self.plot_figure:
            self.plot_fig(X, y_pred)

        return y_pred

if __name__ == "__main__":
    np.random.seed(10)
    num_clusters = 3
    X, _ = make_blobs(n_samples=1000, n_features=2, centers= num_clusters)


    Kmeans = KMeansClustering(X, num_clusters)
    y_pred = Kmeans.fit(X)










































