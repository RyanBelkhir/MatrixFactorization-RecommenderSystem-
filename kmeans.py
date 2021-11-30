import numpy as np

class KMeans():
    """Implementation of K-Means Algorithm"""
    def __init__(self, k_cluster, max_iter=100, n_init=10):
        self.k_cluster = k_cluster
        self.max_iter = max_iter
        self.n_init = n_init

    def update_centroids(self, X, labels):
        """Update the centroids by computing the means of the cluster points"""
        centroids = np.zeros((self.k_cluster, X.shape[1]))
        for k in range(self.k_cluster):
            centroids[k, :] = np.mean(X[labels == k, :], axis=0)

        return centroids

    def distance(self, X, centroids):
        """Compute the distance for every points to every centroids"""
        d = np.zeros((X.shape[0], self.k_cluster))
        for k in range(self.k_cluster):
            d[:, k] = np.square(np.linalg.norm(X - centroids[k, :], axis=1))

        return d

    def score(self, X, labels, centroids):
        """Compute the SSD (sum of squared distance) for the current clusters"""
        d = np.zeros((X.shape[0]))
        for k in range(self.k_cluster):
            d[labels == k] = np.linalg.norm(X[labels == k] - centroids[k, :], axis=1)

        return np.sum(np.square(d))

    def run(self, X):
        """Run 1-iteration of a KMeans with random initialization"""

        permutation = np.random.permutation(np.unique(X, axis=0).shape[0])
        centroids = np.unique(X, axis=0)[permutation[:self.k_cluster]]

        for i in range(self.max_iter):
            old_centroids = centroids
            d = self.distance(X, old_centroids)
            labels = np.argmin(d, axis=1)
            centroids = self.update_centroids(X, labels)
            if np.all(old_centroids == centroids):
                break

        error = self.score(X, labels, centroids)

        return centroids, labels, error

    def fit(self, X, normalize):
        """Run n_init times the K-Means algorithm and take the best one"""
        X = np.copy(X)
        if normalize:
            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0)
            X -= X_mean
            X /= X_std

        best_error = None

        for i in range(self.n_init):
            centroids, labels, error = self.run(X)
            if best_error is None or error < best_error:
                best_centroids = centroids
                best_labels = labels
                best_error = error

        self.centroids = best_centroids
        self.labels = best_labels
        self.error = best_error
        self.compute_clusters()

    def predict(self, X):
        d = self.distance(X, self.centroids)
        return np.argmin(d, axis=1)

    def compute_clusters(self):
        clusters = []
        for k in range(self.k_cluster):
            idx = np.where(self.labels == k)[0]
            clusters.append(list(idx))
        self.clusters = clusters

    def compute_masks_users(self, R):
        """"Compute k_cluster masks to only consider users of a same cluster"""
        masks = []
        size = R.shape
        for k in range(self.k_cluster):
            mask = np.zeros((size))
            for i in self.clusters[k]:
                mask[i, :] = 1

            mask[R == 0] = 0
            masks.append(mask)

        return masks

    def compute_masks_movies(self, R):
        """"Compute k_cluster movies to only consider movies of a same cluster"""
        masks = []
        size = R.shape
        for k in range(self.k_cluster):
            mask = np.zeros((size))
            for j in self.clusters[k]:
                if idx_M_info[str(j)] in idx.keys():
                    col_idx = idx[idx_M_info[str(j)]]
                    mask[:, col_idx] = 1

            mask[R == 0] = 0
            masks.append(mask)

        return masks


def compute_clusters(labels, k_cluster):
    clusters = []
    for k in range(k_cluster):
        idx = np.where(labels == k)[0]
        clusters.append(list(idx))

    return clusters


from sklearn.metrics import silhouette_score


def silhouette_elbow_method(X, list_k):
    """Silhouette and Elbow method to find the best K in K-Means"""
    errors = []
    sil = []
    for k in list_k:
        kmeans = KModes(k)
        kmeans.fit(X)
        errors.append(kmeans.cost_)
        labels = kmeans.labels_
        sil.append(silhouette_score(X, labels, metric='euclidean'))
    # Plot sse against k
    plt.figure(figsize=(6, 6))
    plt.plot(list_k, errors, '-o')
    plt.xlabel(r'Number of clusters *k*')
    plt.ylabel('Sum of squared distance')
    plt.plot()
    plt.figure(figsize=(6, 6))
    plt.plot(list_k, sil, '-o')
    plt.xlabel(r'Number of clusters *k*')
    plt.ylabel("Silhouette's mesure")
