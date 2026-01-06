import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE

# 1. LOAD THE DATA
print("Loading Iris dataset...")
iris = datasets.load_iris()
X = iris.data         # This is the 4-dimensional data
y = iris.target       # This is the "answer" (the 3 species)

# 2. THE ALGORITHM (This is the core of the idea!)
# We are reducing 4 dimensions (X) down to 3 dimensions
print("Running t-SNE... this can take a moment.")
tsne_model = TSNE(n_components=3, # We want a 3D map
                  learning_rate='auto',
                  init='pca', 
                  perplexity=30) # A good default

# This one line runs the whole algorithm!
X_3D = tsne_model.fit_transform(X)

print("Data transformed to 3D!")
print(X_3D.shape) # Should show (150, 3)

# 3. PROVE IT WORKED (Visualize the 3D Map)
print("Generating 3D plot...")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot. We color the dots based on their real species (y)
# This will show us if the *clusters* were preserved
scatter = ax.scatter(X_3D[:, 0],  # The new X
                     X_3D[:, 1],  # The new Y
                     X_3D[:, 2],  # The new Z
                     c=y)         # Color by species

ax.set_title("3D t-SNE Projection of 4D Iris Data")
ax.set_xlabel("t-SNE Component 1")
ax.set_ylabel("t-SNE Component 2")
ax.set_zlabel("t-SNE Component 3")

# Add a legend
legend1 = ax.legend(*scatter.legend_elements(), title="Species")
ax.add_artist(legend1)

plt.show()