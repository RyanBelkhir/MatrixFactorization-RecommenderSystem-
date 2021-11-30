from kmodes.kmodes import KModes
import time
import numpy as np
from optimizer import SGD, ALS, SVD, GradientDescent
import matplotlib.pyplot as plt
from kmeans import compute_clusters
from data_init import U_info, M_info

class Optimizer_clusterize:

    def __init__(self, latent_vector, mode, R, R_test, k_users, k_movies, lambda_reg, epochs):
        self.latent_vector = latent_vector

        # Mode "users" if we want to cluster the users, "movies" to cluster the movies and "both" to cluster
        # movies and users at the same time
        self.mode = mode
        self.R = R
        self.R_test = R_test
        self.k_users = k_users
        self.k_movies = k_movies
        self.lambda_reg = lambda_reg
        self.epochs = epochs
        self.history = {"train": [],
                        "test": [],
                        "time": []}

    def update_time(self, t0):
        """Update the computational time in the history"""
        t1 = time.time()
        self.history["time"].append(t1 - t0)

    def score(self):
        """Compute the RMSE of the model and update the history by checking every submatrix of R according
        the clusters they represent"""
        if (self.mode == 'users' or self.mode == 'movies'):
            val = []
            for k in range(len(self.submatrices_R)):
                R_hat = self.submatrices_U[k] @ self.submatrices_M[k].T
                R_hat = R_hat[self.submatrices_R[k] > 0].flatten()
                R_pos = self.submatrices_R[k][self.submatrices_R[k] > 0].flatten()
                val += np.square(R_pos - R_hat).tolist()
            val = np.array(val)
            err = np.sqrt(val.mean())
            print(err)

            self.history["train"].append(err)

        elif (self.mode == 'both'):
            val = []
            for k in range(len(self.submatrices_R)):
                for k2 in range(len(self.submatrices_R[k])):
                    R_hat = self.submatrices_U[k][k2] @ self.submatrices_M[k][k2].T
                    R_hat = R_hat[self.submatrices_R[k][k2] > 0].flatten()
                    R_pos = self.submatrices_R[k][k2][self.submatrices_R[k][k2] > 0].flatten()
                    val += np.square(R_pos - R_hat).tolist()
            val = np.array(val)
            err = np.sqrt(val.mean())
            print(err)
            self.history["train"].append(err)

        return err

    def test(self):
        """Compute the RMSE on the test set of the model and update the history by checking every submatrix of R
            according the clusters they represent"""
        if (self.mode == 'users' or self.mode == 'movies'):
            val = []
            for k in range(len(self.submatrices_R)):
                R_hat = self.submatrices_U[k] @ self.submatrices_M[k].T
                R_hat = R_hat[self.submatrices_R_test[k] > 0].flatten()
                R_pos = self.submatrices_R_test[k][self.submatrices_R_test[k] > 0].flatten()
                val += np.square(R_pos - R_hat).tolist()
            val = np.array(val)
            err = np.sqrt(val.mean())
            print(err)
            self.history["test"].append(err)

        else:
            val = []
            for k in range(len(self.submatrices_R)):
                for k2 in range(len(self.submatrices_R[k])):
                    R_hat = self.submatrices_U[k][k2] @ self.submatrices_M[k][k2].T
                    R_hat = R_hat[self.submatrices_R_test[k][k2] > 0].flatten()
                    R_pos = self.submatrices_R_test[k][k2][self.submatrices_R_test[k][k2] > 0].flatten()
                    val += np.square(R_pos - R_hat).tolist()
            val = np.array(val)

            err = np.sqrt(val.mean())
            print(err)
            self.history["test"].append(err)

        return err

    def plot_history(self):

        print("train error:", self.history["train"][-1])
        print("test error:", self.history["test"][-1])

        plt.plot(np.arange(0, len(self.history["train"])), self.history["train"], label='train')
        plt.plot(np.arange(0, len(self.history["test"])), self.history["test"], label='test')
        plt.legend()
        plt.title('Precision of the model according to time')
        plt.xlabel('Epochs')
        plt.ylabel('RMSE')
        plt.plot()

    def compute_submatrix(self, cluster_users, cluster_movies, mode):
        """Divide R into submatrix according to users and/or movies clusters"""
        submatrices_R = []
        submatrices_R_test = []
        if (self.mode == 'users'):
            for k in range(self.k_users):
                submatrix_R = self.R[cluster_users[k], :]
                submatrix_R_test = self.R_test[cluster_users[k], :]

                submatrix_R_test = submatrix_R_test[:, ~np.all(submatrix_R == 0, axis=0)]
                submatrix_R = submatrix_R[:, ~np.all(submatrix_R == 0, axis=0)]

                submatrices_R.append(submatrix_R)
                submatrices_R_test.append(submatrix_R_test)

        elif (self.mode == 'movies'):
            for k in range(self.k_movies):
                movies_idx = get_new_id(cluster_movies[k])
                submatrix_R = self.R[:, movies_idx]
                submatrix_R_test = self.R_test[:, movies_idx]

                submatrix_R_test = submatrix_R_test[~np.all(submatrix_R == 0, axis=1)]
                submatrix_R = submatrix_R[~np.all(submatrix_R == 0, axis=1)]

                submatrices_R.append(submatrix_R)
                submatrices_R_test.append(submatrix_R_test)

        else:
            for k_u in range(self.k_users):
                user_submatrices_R = []
                user_submatrices_R_test = []
                for k_m in range(self.k_movies):
                    submatrix_R = self.R[cluster_users[k_u], :]
                    submatrix_R_test = self.R_test[cluster_users[k_u], :]

                    movies_idx = get_new_id(cluster_movies[k_m])
                    submatrix_R = submatrix_R[:, movies_idx]
                    submatrix_R_test = submatrix_R_test[:, movies_idx]

                    submatrix_R_test = submatrix_R_test[~np.all(submatrix_R == 0, axis=1)]
                    submatrix_R = submatrix_R[~np.all(submatrix_R == 0, axis=1)]

                    submatrix_R_test = submatrix_R_test[:, ~np.all(submatrix_R == 0, axis=0)]
                    submatrix_R = submatrix_R[:, ~np.all(submatrix_R == 0, axis=0)]

                    user_submatrices_R.append(submatrix_R)
                    user_submatrices_R_test.append(submatrix_R_test)

                submatrices_R.append(user_submatrices_R)
                submatrices_R_test.append(user_submatrices_R_test)

        self.submatrices_R = submatrices_R
        self.submatrices_R_test = submatrices_R_test

    def init_matrix(self):
        """Randomly initialize U and M matrices for every submatrix R"""
        self.submatrices_U = []
        self.submatrices_M = []
        if (self.mode == 'users'):

            for k in range(self.k_users):
                R = self.submatrices_R[k]
                U = np.random.random((R.shape[0], self.latent_vector))
                M = np.random.random((R.shape[1], self.latent_vector))
                self.submatrices_U.append(U)
                self.submatrices_M.append(M)

        elif (self.mode == 'movies'):

            for k in range(self.k_movies):
                R = self.submatrices_R[k]
                U = np.random.random((R.shape[0], self.latent_vector))
                M = np.random.random((R.shape[1], self.latent_vector))
                self.submatrices_U.append(U)
                self.submatrices_M.append(M)

        else:
            for k in range(self.k_users):
                submatrices_U_Users = []
                submatrices_M_Users = []
                for k2 in range(self.k_movies):
                    R = self.submatrices_R[k][k2]
                    U = np.random.random((R.shape[0], self.latent_vector))
                    M = np.random.random((R.shape[1], self.latent_vector))
                    submatrices_U_Users.append(U)
                    submatrices_M_Users.append(M)
                self.submatrices_U.append(submatrices_U_Users)
                self.submatrices_M.append(submatrices_M_Users)

    def fit(self):
        pass

    def run(self):
        """Create the users and/or movies clusters and compute the submatrix R and U, M for every cluster"""
        if (self.mode == 'users'):
            kmeans_users = KModes(self.k_users)
            kmeans_users.fit(U_info)
            cluster_users = compute_clusters(kmeans_users.labels_, self.k_users)
            self.compute_submatrix(cluster_users, None, mode=self.mode)
            self.init_matrix()

        elif (self.mode == 'movies'):
            kmeans_movies = KModes(self.k_movies, init='Huang', n_init=5, verbose=1)
            kmeans_movies.fit(M_info)
            cluster_movies = compute_clusters(kmeans_movies.labels_, self.k_movies)
            self.compute_submatrix(None, cluster_movies, mode=self.mode)
            self.init_matrix()

        else:
            kmeans_users = KModes(self.k_users)
            kmeans_users.fit(U_info)
            cluster_users = compute_clusters(kmeans_users.labels_, self.k_users)
            kmeans_movies = KModes(self.k_movies)
            kmeans_movies.fit(M_info)
            cluster_movies = compute_clusters(kmeans_movies.labels_, self.k_movies)

            self.compute_submatrix(cluster_users, cluster_movies, mode=self.mode)
            self.init_matrix()


class ALS_clusterize(Optimizer_clusterize):
    """ Clusterize version of the ALS algorithm 
    Select a mode to clusterize between "users", "movies" and "both"
    """
    def __init__(self, latent_vector, mode, R, R_test, k_users=7, k_movies=9, lambda_reg=3.5, epochs=5):
        super().__init__(latent_vector, mode, R, R_test, k_users, k_movies, lambda_reg, epochs)

    def fit(self, U, M):
        self.run()
        self.score()
        self.test()
        if (self.mode == 'users'):

            for _ in range(self.epochs):
                for k in range(self.k_users):
                    R = self.submatrices_R[k]
                    R_test = self.submatrices_R_test[k]
                    U = self.submatrices_U[k]
                    M = self.submatrices_M[k]
                    als = ALS(self.latent_vector, R=R, R_test=R_test, U=U, M=M, lambda_reg=self.lambda_reg, epochs=5)
                    self.submatrices_U[k], self.submatrices_M[k] = als.fit()

                self.score()
                self.test()

        elif (self.mode == 'movies'):

            for _ in range(self.epochs):
                for k in range(self.k_movies):
                    R = self.submatrices_R[k]
                    R_test = self.submatrices_R_test[k]
                    U = self.submatrices_U[k]
                    M = self.submatrices_M[k]
                    als = ALS(self.latent_vector, R=R, R_test=R_test, U=U, M=M, lambda_reg=self.lambda_reg,
                                epochs=self.epochs)
                    self.submatrices_U[k], self.submatrices_M[k] = als.fit()

                self.score()
                self.test()

        else:

            for _ in range(self.epochs):
                for k in range(self.k_users):
                    for k2 in range(self.k_movies):
                        R = self.submatrices_R[k][k2]
                        R_test = self.submatrices_R_test[k][k2]
                        U = self.submatrices_U[k][k2]
                        M = self.submatrices_M[k][k2]
                        als = ALS(self.latent_vector, R=R, R_test=R_test, U=U, M=M, lambda_reg=self.lambda_reg,
                                    epochs=self.epochs)
                        self.submatrices_U[k][k2], self.submatrices_M[k][k2] = als.fit()
                self.score()
                self.test()

                # self.plot_history()


class SGD_clusterize(Optimizer_clusterize):
    """Clusterize version of the SGD algorithm"""

    def __init__(self, lr, latent_vector, mode, R, R_test, k_users=7, k_movies=24, lambda_reg=0.1,
                    epochs=20):
        super().__init__(latent_vector, mode, R, R_test, k_users, k_movies, lambda_reg, epochs)
        self.lr = lr

    def fit(self, U, M):
        t = time.time()
        self.run()
        self.score()
        self.test()
        self.update_time(t)
        if (self.mode == 'users'):
            for _ in range(self.epochs):
                for k in range(self.k_users):
                    R = self.submatrices_R[k]
                    R_test = self.submatrices_R_test[k]
                    U = self.submatrices_U[k]
                    M = self.submatrices_M[k]
                    sgd = SGD(0.001, R=R, R_test=R_test, U=U, M=M, lambda_reg=self.lambda_reg, epochs=4)
                    self.submatrices_U[k], self.submatrices_M[k] = sgd.fit()

                self.score()
                self.test()
                self.update_time(t)

        elif (self.mode == 'movies'):

            for _ in range(self.epochs):
                for k in range(self.k_movies):
                    R = self.submatrices_R[k]
                    R_test = self.submatrices_R_test[k]

                    R_test = R_test[~np.all(R == 0, axis=1)]
                    R = R[~np.all(R == 0, axis=1)]

                    self.submatrices_R_test[k] = self.submatrices_R_test[k][~np.all(self.submatrices_R[k] == 0, axis=1)]
                    self.submatrices_R[k] = self.submatrices_R[k][~np.all(self.submatrices_R[k] == 0, axis=1)]

                    U = self.submatrices_U[k]
                    M = self.submatrices_M[k]
                    sgd = SGD(self.lr, R=R, R_test=R_test, U=U, M=M, lambda_reg=self.lambda_reg, epochs=10)
                    self.submatrices_U[k], self.submatrices_M[k] = sgd.fit()

                self.score()
                self.test()
                self.update_time(t)

        else:

            for _ in range(self.epochs):
                for k in range(self.k_users):
                    for k2 in range(self.k_movies):
                        R = self.submatrices_R[k][k2]
                        R_test = self.submatrices_R_test[k][k2]

                        R_test = R_test[~np.all(R == 0, axis=1)]

                        R = R[~np.all(R == 0, axis=1)]
                        U = self.submatrices_U[k][k2]
                        M = self.submatrices_M[k][k2]
                        sgd = SGD(self.lr, R=R, R_test=R_test, U=U, M=M, lambda_reg=self.lambda_reg, epochs=10)
                        self.submatrices_U[k][k2], self.submatrices_M[k][k2] = sgd.fit()
                self.score()
                self.test()
                self.update_time(t)

        self.plot_history()
        print("time:", self.history["time"][-1])


class GD_clusterize(Optimizer_clusterize):
    """Clusterize version of the Gradient Descent algorithm"""
    def __init__(self, lr, latent_vector, mode, R, R_test, k_users=7, k_movies=19, lambda_reg=0.1,
                    epochs=100):
        super().__init__(latent_vector, mode, R, R_test, k_users, k_movies, lambda_reg, epochs)
        self.lr = lr

    def fit(self, U, M):
        t = time.time()
        self.run()
        self.score()
        self.test()
        self.update_time(t)
        if (self.mode == 'users'):
            for _ in range(self.epochs):
                for k in range(self.k_users):
                    R = self.submatrices_R[k]
                    R_test = self.submatrices_R_test[k]
                    U = self.submatrices_U[k]
                    M = self.submatrices_M[k]
                    gd = GradientDescent(self.lr, R=R, R_test=R_test, U=U, M=M, lambda_reg=self.lambda_reg, epochs=5)
                    self.submatrices_U[k], self.submatrices_M[k] = gd.fit()

                self.score()
                self.test()
                self.update_time(t)

        elif (self.mode == 'movies'):

            for _ in range(self.epochs):
                for k in range(self.k_movies):
                    R = self.submatrices_R[k]
                    R_test = self.submatrices_R_test[k]
                    U = self.submatrices_U[k]
                    M = self.submatrices_M[k]
                    gd = GradientDescent(self.lr, R=R, R_test=R_test, U=U, M=M, lambda_reg=self.lambda_reg, epochs=3)
                    self.submatrices_U[k], self.submatrices_M[k] = gd.fit()

                self.score()
                self.test()
                self.update_time(t)

        else:

            for _ in range(self.epochs):
                for k in range(self.k_users):
                    for k2 in range(self.k_movies):
                        R = self.submatrices_R[k][k2]
                        R_test = self.submatrices_R_test[k][k2]
                        U = self.submatrices_U[k][k2]
                        M = self.submatrices_M[k][k2]
                        gd = GradientDescent(self.lr, R=R, R_test=R_test, U=U, M=M, lambda_reg=self.lambda_reg,
                                                epochs=5)
                        self.submatrices_U[k][k2], self.submatrices_M[k][k2] = gd.fit()
                self.score()
                self.test()
                self.update_time(t)

        # self.plot_history()
        print("time:", self.history["time"][-1])


class SVD_clusterize(Optimizer_clusterize):
    """Clusterize version of the SVD algorithm"""
    def __init__(self, latent_vector, mode, R, R_test, k_users=7, k_movies=19, lambda_reg=0.1, epochs=1):
        super().__init__(latent_vector, mode, R, R_test, k_users, k_movies, lambda_reg, epochs)

    def fit(self):
        t = time.time()
        self.run()
        self.score()
        self.test()
        self.update_time(t)
        if (self.mode == 'users'):
            for _ in range(self.epochs):
                for k in range(self.k_users):
                    R = self.submatrices_R[k]
                    R_test = self.submatrices_R_test[k]

                    svd = SVD(self.latent_vector, R=R, R_test=R_test, lambda_reg=self.lambda_reg, epochs=50)
                    self.submatrices_U[k], self.submatrices_M[k] = svd.fit()

                self.score()
                self.test()
                self.update_time(t)

        elif (self.mode == 'movies'):

            for _ in range(self.epochs):
                for k in range(self.k_movies):
                    R = self.submatrices_R[k]
                    R_test = self.submatrices_R_test[k]

                    svd = SVD(self.latent_vector, R=R, R_test=R_test, lambda_reg=self.lambda_reg, epochs=50)
                    self.submatrices_U[k], self.submatrices_M[k] = svd.fit()

                self.score()
                self.test()
                self.update_time(t)

        else:

            for _ in range(self.epochs):
                for k in range(self.k_users):
                    for k2 in range(self.k_movies):
                        R = self.submatrices_R[k][k2]
                        R_test = self.submatrices_R_test[k][k2]

                        svd = SVD(self.latent_vector, R=R, R_test=R_test, lambda_reg=self.lambda_reg, epochs=50)
                        self.submatrices_U[k][k2], self.submatrices_M[k][k2] = svd.fit()
                self.score()
                self.test()
                self.update_time(t)

        self.plot_history()
        print("time:", self.history["time"][-1])
