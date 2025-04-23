import random
import numpy as np


matrix = np.random.randint(0, 256, (lambda n=random.randint(1, 5): (n, n))())
file_path_ratings = '../data/ratings.csv'
file_path_movies = '../data/movies.csv'
