import numpy as np
import matplotlib.pyplot as plt

class KMeansClustering:
    def __init__(self, k, max_iterations=10000, random_seed=42):
        """
        Initialize the KMeansClustering instance.

        Parameters:
            k (int): Number of clusters.
            max_iterations (int): Maximum number of iterations for K-means.
            random_seed (int): Seed for random initialization of centroids to be recallable with the same value.

        Example:
        >>> kmeans = KMeansClustering(k=3)
        """
        self.k = k
        self.max_iterations = max_iterations
        self.random_seed = random_seed
        self.centroids = None
        self.labels = None

    def fit(self, X):
        """
        Fit the K-means clustering model to the input data.

        Parameters:
            X (numpy.ndarray): Input data.

        Returns:
            None

        Example:
        >>> X_train = np.random.rand(100, 2)
        >>> kmeans.fit(X_train)
        """
        np.random.seed(self.random_seed)
        self.centroids = X[np.random.choice(len(X), self.k, replace=False)]

        for _ in range(self.max_iterations):
            #Calculate distances between data points and centroids
            distances = np.sqrt(np.sum((X[:, np.newaxis] - self.centroids) ** 2, axis=2))

            #Assign each data point to the closest centroid
            self.labels = np.argmin(distances, axis=1)

            #Update centroids
            new_centroids = np.array([X[self.labels == idx].mean(axis=0) for idx in range(self.k)])

            #Check for convergence
            if np.allclose(new_centroids, self.centroids):
                break

            self.centroids = new_centroids

    def plot_clusters(self, X, title='K-Means Clustering', xlabel='Principal Component 1', ylabel='Principal Component 2'):
        """
        Plot the clustered data points along with centroids.

        Parameters:
            X (numpy.ndarray): Input data.
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.

        Returns:
            None, plots a scatterplot with labelled individuals.

        Example:
        >>> kmeans.plot_clusters(X_train, title='K-Means Clustering', xlabel='Feature 1', ylabel='Feature 2')
        """
        #Create a 2D scatter plot
        plt.figure(figsize=(8, 8))

        #Scatter plot for the first two principal components
        plt.scatter(X[:, 0], X[:, 1], c=self.labels, cmap='viridis', edgecolors='k', s=50)

        #Plot centroids
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', marker='X', s=200, label='Centroids')

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()

    def silhouette_score(self, X):
        """
        Calculate the silhouette score for the clustering.

        Parameters:
            X (numpy.ndarray): Input data.

        Returns:
            float: Silhouette score.

        Example:
        >>> silhouette = kmeans.silhouette_score(X_train)
        """
        num_points = len(X)
        a = np.zeros(num_points)
        b = np.zeros(num_points)

        for i in range(num_points):
            cluster_id = self.labels[i]
            cluster_points = X[self.labels == cluster_id]
            a[i] = np.mean(np.sqrt(np.sum((X[i] - cluster_points) ** 2, axis=1)))

            b[i] = np.min([np.mean(np.sqrt(np.sum((X[i] - X[self.labels == other_cluster]) ** 2, axis=1))) for other_cluster in np.unique(self.labels) if other_cluster != cluster_id])
        
        #Silhouette Calculation
        silhouette_values = (b - a) / np.maximum(a, b)
        return np.mean(silhouette_values)

    def elbow_method(self, X, possible_k_values=None):
        """
        Plot the elbow curve to find the optimal number of clusters.

        Parameters:
            X (numpy.ndarray): Input data.
            possible_k_values (list): List of possible k values for the elbow method.

        Returns:
            None, outputs plot of elbow curve.

        Example:
        >>> kmeans.elbow_method(X_train, possible_k_values=[2, 3, 4, 5])
        """
        #Testing possible k values with sum of squared distances
        if possible_k_values is None:
            possible_k_values = range(1, 11)

        ssd = []
        for k in possible_k_values:
            np.random.seed(42)
            centroids = X[np.random.choice(len(X), k, replace=False)]

            for _ in range(self.max_iterations):
                distances = np.sqrt(np.sum((X[:, np.newaxis] - centroids) ** 2, axis=2))
                labels = np.argmin(distances, axis=1)
                new_centroids = np.array([X[labels == idx].mean(axis=0) for idx in range(k)])

                if np.allclose(new_centroids, centroids):
                    break

                centroids = new_centroids

            ssd.append(np.sum((X - centroids[labels]) ** 2))

        #Plot the elbow curve
        plt.figure(figsize=(8, 6))
        plt.plot(possible_k_values, ssd, marker='o')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Sum of Squared Distances (SSD)')
        plt.show()
