import numpy as np
import time
import matplotlib.pyplot as plt
import numba as nb 




class Optimizer:
    def __init__(self, U, M, R, R_test, lambda_reg, epochs):
        self.U = np.copy(U)
        self.M = np.copy(M)
        self.R = np.copy(R)
        self.R_test = np.copy(R_test)
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
        """Compute the RMSE of the model and update the history"""
        R_hat = self.U @ self.M.T
        R_hat = R_hat[self.R > 0]
        R_pos = self.R[self.R > 0]
        err = np.sqrt(np.square(R_pos - R_hat).mean())
        self.history["train"].append(err)
        return err

    def test(self):
        """Compute the RMSE of the model on the test set and update the history"""
        R_hat = self.U @ self.M.T
        R_hat = R_hat[self.R_test > 0]
        R_pos = self.R_test[self.R_test > 0]
        err = np.sqrt(np.square(R_pos - R_hat).mean())
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

    def fit(self):
        pass

    # About cross validation

    def k_folds(self, Ratings, k=10):
        """Returns K-Folds with at least one rating for every user"""
        R_copy = np.copy(Ratings)
        n, m = R_copy.shape
        folds = []
        for _ in range(k):
            fold = np.zeros((n, m))
            for i in range(n):
                cols = np.where(R_copy[i, :] > 0)[0]
                j = np.random.choice(cols)
                fold[i][j] = R_copy[i][j]
                R_copy[i][j] = 0
            folds.append(fold)

        col, row = np.where(R_copy)
        coord = np.array(list((zip(col, row))))
        np.random.shuffle(coord)
        data_split = np.array_split(coord, k)

        for idx, data in enumerate(data_split):
            for i, j in data:
                folds[idx][i, j] = R_copy[i, j]
                R_copy[i, j] = 0

        return folds

    def train_test_folds(self, folds, idx):
        """Return the idx Flods as a test set and combine the others as a train set"""
        train = np.zeros((n, m))
        test = np.zeros((n, m))

        for i in range(len(folds)):
            if i != idx:
                train += folds[i]
            else:
                test = folds[i]
        return train, test

    def cross_validation(R, parameters, k=10):
        pass


class GradientDescent(Optimizer):
    def __init__(self, lr, U, M, R, R_test, lambda_reg=0.002, epochs=20):
        super().__init__(U, M, R, R_test, lambda_reg, epochs)
        self.lr = lr
        self.best_param = {"learning_rate": 0,
                            "lambda": 0,
                            "K": 0}
        self.best_rmse = None

    def fit(self):
        t = time.time()
        self.score()
        self.test()

        for epoch in range(self.epochs):
            R_hat = self.U @ self.M.T
            # R_hat = np.einsum("ik,jk->ij", U, M)
            R_hat[self.R <= 0] = 0
            E = self.R - R_hat

            U_temp = self.U + self.lr * (2 * E @ self.M - 2 * self.lambda_reg * self.U)
            M_temp = self.M + self.lr * (2 * E.T @ self.U - 2 * self.lambda_reg * self.M)
            self.U = U_temp
            self.M = M_temp

            self.score()
            self.test()
            self.update_time(t)

        # self.plot_history()
        return self.U, self.M

    # About cross validation

    def cross_validation(self, Ratings, parameters, k=10):
        folds = self.k_folds(Ratings, k)
        best_param = {"learning_rate": 0,
                        "lambda": 0,
                        "K": 0}
        best_rmse = None

        for latent_vector in parameters['K']:
            U = np.random.random(size=(n, latent_vector))
            M = np.random.random(size=(m, latent_vector))
            for lr in parameters['learning_rate']:
                for reg in parameters['lambda']:
                    errors = []
                    for i in range(k):
                        R_tr, R_te = self.train_test_folds(folds, i)
                        gd_cv = GradientDescent(lr, U, M, R_tr, R_te, lambda_reg=reg)
                        gd_cv.fit()
                        e = gd_cv.test()
                        errors.append(e)

                    if best_rmse is None or np.mean(errors) < best_rmse:
                        best_rmse = np.mean(errors)
                        best_param["learning_rate"] = lr
                        best_param["lambda"] = reg
                        best_param["K"] = latent_vector

        self.best_rmse = best_rmse
        self.best_param = best_param
        return best_rmse, best_param


from numba import njit


@njit
def compute_sgd(R, U, M, S, lambda_reg, lr):
    """Compute a SGD iteration with numba to accelerate the process"""
    for i, j in S:
        eij = R[i][j] - (U[i][:] @ M[j][:])
        ui_temp = U[i][:] + lr * (eij * M[j][:] - lambda_reg * U[i][:])
        mj_temp = M[j][:] + lr * (eij * U[i][:] - lambda_reg * M[j][:])

        U[i][:], M[j][:] = ui_temp, mj_temp

    return U, M


class SGD(Optimizer):
    def __init__(self, lr, U, M, R, R_test, lambda_reg=0.1, epochs=20):
        super().__init__(U, M, R, R_test, lambda_reg, epochs)
        self.lr = lr

    def fit(self):
        n, m = self.R.shape
        t = time.time()
        self.score()
        self.test()

        row, col = np.where(self.R > 0)
        S = nb.typed.List(list(zip(row, col)))

        for _ in range(self.epochs):
            np.random.shuffle(S)

            R = self.R
            U = self.U
            M = self.M
            lambda_reg = self.lambda_reg
            lr = self.lr

            self.U, self.M = compute_sgd(R, U, M, S, lambda_reg, lr)

            self.score()
            self.test()
            self.update_time(t)
        self.plot_history()

        return self.U, self.M


class ALS(Optimizer):
    def __init__(self, latent_vector, U, M, R, R_test, lambda_reg=1, epochs=5):
        super().__init__(U, M, R, R_test, lambda_reg, epochs)
        self.latent_vector = latent_vector

    def fit(self):
        self.score()
        self.test()

        n, m = self.R.shape

        # Create a mask W to only consider the ratings > 0
        W = np.zeros((n, m))
        W[self.R > 0] = 1

        for step in range(self.epochs):
            for i in range(n):
                M_temp = np.multiply(self.M.T, W[i, :])
                self.U[i, :] = np.linalg.inv(
                    M_temp @ self.M + self.lambda_reg * np.eye(self.latent_vector)) @ M_temp @ self.R[i, :]
            for j in range(m):
                U_temp = np.multiply(self.U.T, W[:, j])
                self.M[j, :] = np.linalg.inv(
                    U_temp @ self.U + self.lambda_reg * np.eye(self.latent_vector)) @ U_temp @ self.R[:, j]

            self.score()
            self.test()

        # self.plot_history()
        return self.U, self.M

    def fit_clusterize(self):
        """Compute an ALS method where the mask also consider the users and/or movies of a same cluster"""
        self.score()
        self.test()

        n, m = self.R.shape
        kmeans_users = KMeans(7)
        kmeans_users.fit(U_info)
        masks_users = kmeans_users.compute_masks_users(self.R)

        kmeans_movies = KMeans(22)
        kmeans_movies.fit(M_info)
        masks_movies = kmeans_movies.compute_masks_movies(self.R)
        # W = np.zeros((n,m))
        # W[self.R>0] = 1

        for step in range(self.epochs):
            for i in range(n):
                cluster_idx = kmeans_users.labels[i]
                mask = masks_users[cluster_idx]
                M_temp = np.multiply(self.M.T, mask[i, :])
                self.U[i, :] = np.linalg.inv(
                    M_temp @ self.M + self.lambda_reg * np.eye(self.latent_vector)) @ M_temp @ self.R[i, :]
            for j in range(m):
                col_idx = rev_idx[str(j)]

                cluster_idx = kmeans_movies.labels[int(col_idx)]
                mask = masks_movies[cluster_idx]
                U_temp = np.multiply(self.U.T, mask[:, j])
                self.M[j, :] = np.linalg.inv(
                    U_temp @ self.U + self.lambda_reg * np.eye(self.latent_vector)) @ U_temp @ self.R[:, j]

            self.score()
            self.test()

        self.plot_history()

        return self.U, self.M

    def cross_validation(self, Ratings, parameters, k=10):
        folds = self.k_folds(Ratings, k)
        best_param = {
            "lambda": 0,
            "K": 0}
        best_rmse = None

        for latent_vector in parameters['K']:
            U = np.random.random(size=(n, latent_vector))
            M = np.random.random(size=(m, latent_vector))
            for reg in parameters['lambda']:
                errors = []
                for i in range(k):
                    R_tr, R_te = self.train_test_folds(folds, i)
                    als_cv = ALS(lr, U, M, R_tr, R_te, lambda_reg=reg)
                    als_cv.fit()
                    e = als_cv.test()
                    errors.append(e)
                if best_rmse is None or np.mean(errors) < best_rmse:
                    best_rmse = np.mean(errors)
                    best_param["lambda"] = reg
                    best_param["K"] = latent_vector

        return best_rmse, best_param


class SVD(Optimizer):
    def __init__(self, latent_vector, U, M, R, R_test, lambda_reg=None, epochs=50):
        super().__init__(U, M, R, R_test, lambda_reg, epochs)
        self.latent_vector = latent_vector
        self.similarity_U = None
        self.similarity_M = None

    def fit(self):
        """Compute a SVD that take in consideration the missing entries by replacing them by the means of the user's rate
        according to the book
        """
        t = time.time()
        Rf = np.copy(self.R)
        n, m = Rf.shape

        # Initialisation

        mean_user = self.R.sum(1) / np.sum(self.R != 0, axis=1)
        for i in range(n):
            pos = np.where(self.R[i] == 0)
            Rf[i, pos] = mean_user[i]

        # Iterative steps
        for _ in range(self.epochs):
            Q, E, P = extmath.randomized_svd(Rf, self.latent_vector)

            self.U = Q @ np.diag(E)
            self.M = (P).T

            Rf = self.U @ self.M.T
            # We replace originally missing values by values in Rf
            Rf[np.where(self.R != 0)] = 0
            Rf += self.R

            self.score()
            self.test()
            self.update_time(t)

        # self.plot_history()
        return self.U, self.M

    def GD_ALS_method(self, lr, method):
        """ ALS or Gradient Descent methods with an SVD initialization"""
        t = time.time()
        Rf = np.copy(self.R)
        n, m = Rf.shape

        # Initialisation
        moy_user = self.R.sum(1) / np.sum(self.R != 0, axis=1)
        for i in range(n):
            pos = np.where(self.R[i] == 0)
            Rf[i, pos] = moy_user[i]

        Q, E, P = extmath.randomized_svd(Rf, self.latent_vector)
        self.U = Q @ np.diag(E)
        self.M = (P).T
        Rf = self.U @ self.M.T

        # Gradient descent
        if method == 'ALS':
            svd_method = ALS(self.latent_vector, self.U, self.M, R_train, R_test, lambda_reg=1, epochs=5)
        else:
            svd_method = GradientDescent(lr, self.U, self.M, R_train, R_test, lambda_reg=0.002, epochs=200)
        svd_method.fit()

        return svd_method.U, svd_method.M

    def cross_validation(self, parameters, k):
        folds = self.k_folds(Ratings, k)
        best_param = {
            "K": 0}
        best_rmse = None

        for latent_vector in parameters['K']:
            U = np.random.random(size=(n, latent_vector))
            M = np.random.random(size=(m, latent_vector))
            errors = []
            for i in range(k):
                R_tr, R_te = self.train_test_folds(folds, i)
                svd_cv = SVD(latent_vector)
                svd_cv.fit()
                e = svd_cv.test()
                errors.append(e)
            if best_rmse is None or np.mean(errors) < best_rmse:
                best_rmse = np.mean(errors)
                best_param["K"] = latent_vector

        return best_rmse, best_param
