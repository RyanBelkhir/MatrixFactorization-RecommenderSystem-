import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt


#Splitting into train / test set
def train_test(R):
    R_train = np.copy(R)
    R_test = np.zeros((n, m))
    for i in range(n):
        cols = np.where(R_train[i, :] > 0)[0]
        j = np.random.choice(cols)
        R_test[i][j] = R_train[i][j]
        R_train[i][j] = 0

    return R_train, R_test


#Initialization of R (matrix of ratings)
def fullfill_matrix():
    R = np.zeros((n, m))
    for l in range(rates):
        i = df['userId'].iloc[l] - 1
        movie_id = df['movieId'].iloc[l]
        j = idx[movie_id]
        rij = df.iloc[l, 2]

        R[i][j] = rij

    return R


#Import the data
print("Importing ratings...")
df = pd.read_csv('data/ratings.dat', sep="::",header=None, engine='python')
df = df.rename(columns={0: 'userId', 1:'movieId', 2:'rating', 3:'timestamp'})


#Convert it to numpy
users = df['userId'].to_numpy()
movies = df['movieId'].to_numpy()
ratings = df['rating'].to_numpy()

# Delete users that have exactly 0 or 1 rating
R = csr_matrix((ratings, (users,movies))).toarray()
rates_per_user = (R != 0).sum(axis=1)
R = R[rates_per_user>1]
print("Ratings downloaded.")
#Initialize the size
n, m = R.shape
rates = len(df['timestamp'])

print("Importing movies...")
#Iniatialize matrix M_info to have Monvie's information in a categorical way
df_genre = pd.read_csv('data/movies.dat', sep="::",header=None, encoding="ISO-8859-1", engine="python")
df_genre = df_genre.rename(columns = {0: 'id' ,1: 'title', 2: 'genre'})


df_title = pd.DataFrame(df_genre.title.str.rsplit('(',1).tolist())
df_title = df_title.rename(columns = {0: 'title' , 1: 'year'})

df_date = pd.DataFrame(df_title.year.str.rsplit(')',1).tolist())

df_movies = pd.concat([df_title['title'], df_date[0]], axis=1, ignore_index=True)
df_movies = pd.concat([df_movies, df_genre['genre']], axis=1, ignore_index=True)

n, _ = df_movies.shape
genres = np.zeros((n, 18))
for i in range (n):
    if 'Action' in df_movies[2][i]:
        genres[i][0] = 1
    if 'Adventure' in df_movies[2][i]:
        genres[i][1] = 1
    if 'Animation' in df_movies[2][i]:
        genres[i][2] = 1
    if "Children" in df_movies[2][i]:
        genres[i][3] = 1
    if 'Comedy' in df_movies[2][i]:
        genres[i][4] = 1
    if 'Crime' in df_movies[2][i]:
        genres[i][5] = 1
    if 'Documentary' in df_movies[2][i]:
        genres[i][6] = 1
    if 'Drama' in df_movies[2][i]:
        genres[i][7] = 1
    if 'Fantasy' in df_movies[2][i]:
        genres[i][8] = 1
    if 'Film-Noir' in df_movies[2][i]:
        genres[i][9] = 1
    if 'Horror' in df_movies[2][i]:
        genres[i][10] = 1
    if 'Musical' in df_movies[2][i]:
        genres[i][11] = 1
    if 'Mystery' in df_movies[2][i]:
        genres[i][12] = 1
    if 'Romance' in df_movies[2][i]:
        genres[i][13] = 1
    if 'Sci-Fi' in df_movies[2][i]:
        genres[i][14] = 1
    if 'Thriller' in df_movies[2][i]:
        genres[i][15] = 1
    if 'War' in df_movies[2][i]:
        genres[i][16] = 1
    if 'Western' in df_movies[2][i]:
        genres[i][17] = 1

df_movies = df_movies.drop(0, axis=1)
df_allgenres = pd.DataFrame(genres)
df_movies = pd.concat([df_movies, df_allgenres], axis=1, ignore_index=True)
df_movies_temp = df_movies.drop(0, axis=1)
df_movies_temp = df_movies_temp.drop(1, axis=1)

M_info = df_movies_temp.to_numpy()
print("Movies imported.")
#Same for the User's informations
print("Importing users data...")
df_users = pd.read_csv('data/users.dat', sep="::",header=None, engine='python')
df_users = df_users.drop(0, axis=1)

df_users = df_users.drop(4, axis=1)
U_info = df_users.to_numpy()
U_info[U_info=='F'] = 0
U_info[U_info=='M'] = 1
print("Users data downloaded.")

#We created a bijection between the movie's id and the column number in the R matrix
idx = {}
new_id = 0
for movie_id in list(df['movieId']):
    if movie_id not in idx.keys():
        idx[movie_id] = new_id
        new_id += 1

idx_M_info = {}
new_id = 0

for movie_id in list(df_genre['id']):
    if new_id not in idx_M_info.keys():
        idx_M_info[new_id] = movie_id
        new_id += 1

rev_idx = {}
for key in list(idx_M_info.keys()):
    if idx_M_info[key] in idx.keys():
        val = idx[idx_M_info[key]]
        rev_idx[val] = key

#Function to get the new id of a movie
def get_new_id(old_idx):
    new_idx = []
    for id in old_idx:
        if idx_M_info[id] in idx.keys():
            new_id = idx[idx_M_info[id]]
            new_idx.append(new_id)

    return new_idx


#Hyperparameters settings
n = len(set(df['userId']))
m = len(set(df['movieId']))
K = 6
rates = len(df)




R = fullfill_matrix()

print("R =", R)
R_train, R_test = train_test(R)
#plot of R
plt.figure(figsize=(50,50))
plt.spy(R, markersize=1)
plt.show()



#Random initialization of U (User's matrix) and M (Movie's matrix)
U = np.random.random(size = (n, K))
M = np.random.random(size = (m, K))