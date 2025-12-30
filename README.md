# PCA from Scratch (NumPy)

Implements Principal Component Analysis using only NumPy:
- standardization
- covariance computation
- eigen decomposition
- explained variance ratio
- projection to k dimensions

## Run
```bash
pip install -r requirements.txt
python pca_numpy.py --in iris_features.csv --k 2 --out iris_pca2.csv
