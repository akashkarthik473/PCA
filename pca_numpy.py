import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

data_array = np.loadtxt('iris.csv', delimiter=',', dtype=float, skiprows=1)
# print(data_array)
feature_means = np.mean(data_array, axis=0)
feature_stds = np.std(data_array, axis=0)
standardized_data = (data_array - feature_means) / feature_stds
# print(standardized_data)
covariance_matrix = np.cov(standardized_data, rowvar=False)
# print(covariance_matrix)

eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
eigenvalues = eigenvalues[::-1]
eigenvectors = eigenvectors[:, ::-1]
# print(eigenvalues)
# print(eigenvectors)
ratio = eigenvalues / np.sum(eigenvalues)
print(ratio)
k = 2
projected_data = standardized_data @ eigenvectors[:, :k]
print(projected_data)

x = projected_data[:, 0]
y = projected_data[:, 1]
# Create species labels: 0=setosa (first 50), 1=versicolor (next 50), 2=virginica (last 50)
species = np.array([0]*50 + [1]*50 + [2]*50)
plt.scatter(x, y, c=species, cmap='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of Iris Dataset')
plt.colorbar(label='Species')
plt.savefig('pca_plot.png', dpi=150)
print("Plot saved to pca_plot.png")

print(np.corrcoef(x, y))