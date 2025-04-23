# Movie Recommendation System with SVD

This repository implements a movie recommendation system using Singular Value Decomposition (SVD), along with 3D data visualization to explore latent features of users and movies.

---

## Project Structure

```
svd-ivelmakina/
├── data/                        # MovieLens dataset (ratings, movies)
│   ├── movies.csv
│   └── ratings.csv
│
├── notebooks/                  # Exploratory notebooks for analysis
│   ├── recommendation_algorithm_selection_of_movies.ipynb
│   └── recommendation_algorithm_visualization.ipynb
│
├── src/                        # Core Python scripts
│   ├── svd_implementation.py   # Manual SVD
│   └── objects.py              # File path configs
│
├── .gitignore
├── .gitattributes
├── README.md
└── requirements.txt
```

---

## Part One – Manual SVD Implementation

### Features

- Implement SVD from scratch using NumPy (excluding `np.linalg.svd`)
- Return matrices U, Σ, and Vᵀ
- Validate decomposition by reconstructing the original matrix: A ≈ U * Σ * Vᵀ

### Resources

- [Video Tutorial 1](https://youtu.be/vSczTbgc8Rc?si=NX8eTqH1KsUmJnZj)
- [Video Tutorial 2](https://youtu.be/mBcLRGuAFUk?si=nxhLO82Zn8-PKRHJ)

---

## Part Two – SVD & 3D Visualization

### Features

- Load and clean the MovieLens dataset
- Filter users with fewer than 200 ratings and movies with fewer than 100 ratings
- Fill missing values with a default (e.g., 2.5)
- Apply SVD on the demeaned matrix
- Visualize users and movies in 3D latent space

### Additional Reading

- [SVD Theory and Algorithm](https://jaketae.github.io/study/svd/)

---

## Part Three – Recommendation Engine

### Features

- Generate predicted ratings using SVD
- Recommend top 10 movies for any user
- Use matrix completion based on predicted scores

### Additional Reading

- [SVD in Recommender Systems (Medium)](https://medium.com/@ritik_gupta/how-singular-value-decomposition-svd-is-used-in-recommendation-systems-clearly-explained-201b24e175db)

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/iravelmakina/svd.git
cd svd

# Set up environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

You can now explore the notebooks or run `svd_implementation.py`.

---

## Dependencies

- Python 3.x
- numpy
- scipy
- pandas
- matplotlib
- IPython
- jupyter

---

## License

This project is open-source under the **MIT License**.

---

## Contributor

- [@iravelmakina](https://github.com/iravelmakina)
