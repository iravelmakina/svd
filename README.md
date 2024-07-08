# Movie Recommendation System

This repository contains a Python program for implementing a movie recommendation system using Singular Value Decomposition (SVD) and data visualization techniques.

## Part One

### Features
- **Implementation of SVD**
  - Write a function that takes any matrix and returns the matrices **U**, **Σ**, and **V^T** using NumPy (excluding built-in functions like `np.linalg.svd`).
  - Verify the decomposition by multiplying the matrices and comparing with the original matrix.
  - Recommended video tutorials: [Video 1](https://youtu.be/vSczTbgc8Rc?si=NX8eTqH1KsUmJnZj), [Video 2](https://youtu.be/mBcLRGuAFUk?si=nxhLO82Zn8-PKRHJ).

### Usage Instructions
1. Implement the SVD function.
2. Verify the SVD by reconstructing the original matrix.

## Part Two

### Features
- **Data Loading and Preprocessing**
  - Load the MovieLens dataset, containing around 100,000 ratings from 600 users on 9,000 movies.
  - Convert the dataset into a Pandas DataFrame for ease of manipulation.
  - Clean the data by removing users who have rated fewer than 200 movies and movies with fewer than 100 ratings.
  - Fill missing ratings with a value (e.g., 2.5).

- **Dimensionality Reduction and Visualization**
  - Perform SVD on the demeaned ratings matrix.
  - Visualize users and movies in 3D latent feature space to observe similarities.
  - Recommended libraries: NumPy, SciPy, Pandas, MatPlotLib.
  - Useful link: [Theory and Algorithm](https://jaketae.github.io/study/svd/).

### Usage Instructions
1. Load the MovieLens dataset and preprocess it.
2. Perform SVD on the demeaned ratings matrix.
3. Visualize users and movies in 3D space.

## Part Three

### Features
- **Recommendation Algorithm**
  - Perform SVD on the data.
  - Generate predicted ratings for all users.
  - Create a function to recommend top 10 movies for any user based on predicted ratings.
  - Recommended library: SciPy.
  - Useful link: [Algorithm Explanation](https://medium.com/@ritik_gupta/how-singular-value-decomposition-svd-is-used-in-recommendation-systems-clearly-explained-201b24e175db).

### Usage Instructions
1. Perform SVD on the ratings matrix.
2. Generate predicted ratings.
3. Implement a function to recommend movies based on predicted ratings.
4. Test the function by generating recommendations for specific users.

## Dependencies

- Python 3.x
- NumPy
- SciPy
- Pandas
- MatPlotLib
- Jupyter Notebook
- IPython

## Contributors

- @iravelmakina
