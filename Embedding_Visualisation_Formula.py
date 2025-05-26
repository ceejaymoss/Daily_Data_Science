import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import gensim.downloader as api
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import time

# Set random seed for reproducibility
np.random.seed(42)

def load_embeddings():
    """Load pre-trained word embeddings from gensim"""
    print("Loading word embeddings...")
    # Load GloVe word vectors (smaller model for demonstration)
    word_vectors = api.load("glove-wiki-gigaword-50")
    
    # Select specific word categories for better visualization
    categories = {
        'animals': ['cat', 'dog', 'horse', 'elephant', 'lion', 'tiger', 'bear', 'monkey', 'zebra', 'giraffe'],
        'fruits': ['apple', 'orange', 'banana', 'grape', 'pear', 'peach', 'strawberry', 'blueberry', 'mango', 'pineapple'],
        'countries': ['usa', 'france', 'germany', 'japan', 'china', 'russia', 'brazil', 'india', 'canada', 'australia'],
        'colors': ['red', 'blue', 'green', 'yellow', 'black', 'white', 'purple', 'orange', 'pink', 'brown'],
        'technology': ['computer', 'phone', 'internet', 'software', 'hardware', 'algorithm', 'data', 'network', 'cloud', 'digital']
    }
    
    # Create a dictionary of word vectors and their categories
    vectors = []
    words = []
    word_categories = []
    
    for category, category_words in categories.items():
        for word in category_words:
            if word in word_vectors:
                vectors.append(word_vectors[word])
                words.append(word)
                word_categories.append(category)
    
    # Convert to numpy arrays
    X = np.array(vectors)
    
    return X, words, word_categories

def apply_pca(X, n_components=2):
    """Apply PCA to reduce dimensionality"""
    print("Applying PCA...")
    start_time = time.time()
    pca = PCA(n_components=n_components)
    result = pca.fit_transform(X)
    print(f"PCA completed in {time.time() - start_time:.2f} seconds")
    explained_variance = pca.explained_variance_ratio_.sum() * 100
    print(f"Explained variance: {explained_variance:.2f}%")
    return result

def apply_tsne(X, n_components=2, perplexity=30):
    """Apply t-SNE to reduce dimensionality"""
    print("Applying t-SNE...")
    start_time = time.time()
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    result = tsne.fit_transform(X)
    print(f"t-SNE completed in {time.time() - start_time:.2f} seconds")
    return result

def apply_umap(X, n_components=2, n_neighbors=15, min_dist=0.1):
    """Apply UMAP to reduce dimensionality"""
    print("Applying UMAP...")
    start_time = time.time()
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, 
                        min_dist=min_dist, random_state=42)
    result = reducer.fit_transform(X)
    print(f"UMAP completed in {time.time() - start_time:.2f} seconds")
    return result

def plot_embeddings(embeddings_2d, words, categories, title, figsize=(12, 10)):
    """Plot 2D embeddings with category colors and word labels"""
    plt.figure(figsize=figsize)
    
    # Create a colormap for categories
    unique_categories = list(set(categories))
    colors = cm.rainbow(np.linspace(0, 1, len(unique_categories)))
    color_map = dict(zip(unique_categories, colors))
    
    # Plot points
    for i, (x, y) in enumerate(embeddings_2d):
        category = categories[i]
        plt.scatter(x, y, color=color_map[category], alpha=0.7)
        plt.annotate(words[i], (x, y), fontsize=9, alpha=0.8)
    
    # Create legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 label=cat, markerfacecolor=color_map[cat], markersize=10)
                      for cat in unique_categories]
    plt.legend(handles=legend_elements, loc='best')
    
    plt.title(title)
    plt.tight_layout()
    return plt

def plot_embeddings_3d(embeddings_3d, words, categories, title, figsize=(14, 12)):
    """Plot 3D embeddings with category colors and word labels"""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a colormap for categories
    unique_categories = list(set(categories))
    colors = cm.rainbow(np.linspace(0, 1, len(unique_categories)))
    color_map = dict(zip(unique_categories, colors))
    
    # Plot points
    for i, (x, y, z) in enumerate(embeddings_3d):
        category = categories[i]
        ax.scatter(x, y, z, color=color_map[category], alpha=0.7)
        ax.text(x, y, z, words[i], size=9, alpha=0.8)
    
    # Create legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 label=cat, markerfacecolor=color_map[cat], markersize=10)
                      for cat in unique_categories]
    ax.legend(handles=legend_elements, loc='best')
    
    ax.set_title(title)
    plt.tight_layout()
    return plt

def compare_visualization_techniques():
    """Compare different visualization techniques for word embeddings"""
    # Load embeddings
    X, words, categories = load_embeddings()
    
    # Apply dimensionality reduction techniques
    pca_2d = apply_pca(X, n_components=2)
    tsne_2d = apply_tsne(X, n_components=2, perplexity=5)
    umap_2d = apply_umap(X, n_components=2, n_neighbors=10, min_dist=0.1)
    
    # 3D visualizations
    pca_3d = apply_pca(X, n_components=3)
    umap_3d = apply_umap(X, n_components=3, n_neighbors=10, min_dist=0.1)
    
    # Plot 2D visualizations
    plot_embeddings(pca_2d, words, categories, "PCA (2D)")
    plt.savefig("pca_2d_visualization.png")
    
    plot_embeddings(tsne_2d, words, categories, "t-SNE (2D)")
    plt.savefig("tsne_2d_visualization.png")
    
    plot_embeddings(umap_2d, words, categories, "UMAP (2D)")
    plt.savefig("umap_2d_visualization.png")
    
    # Plot 3D visualization
    plot_embeddings_3d(pca_3d, words, categories, "PCA (3D)")
    plt.savefig("pca_3d_visualization.png")
    
    plot_embeddings_3d(umap_3d, words, categories, "UMAP (3D)")
    plt.savefig("umap_3d_visualization.png")
    
    # Compare results
    compare_results(pca_2d, tsne_2d, umap_2d, words, categories)
    
    print("Visualization complete. All plots saved.")

def calculate_cluster_metrics(embeddings_2d, categories):
    """Calculate simple cluster quality metrics"""
    # Convert categories to numerical values
    unique_categories = list(set(categories))
    category_to_id = {cat: i for i, cat in enumerate(unique_categories)}
    category_ids = np.array([category_to_id[cat] for cat in categories])
    
    # Calculate within-category distances
    within_distances = []
    for cat_id in range(len(unique_categories)):
        cat_points = embeddings_2d[category_ids == cat_id]
        if len(cat_points) > 1:
            # Calculate average pairwise distance within category
            dists = []
            for i in range(len(cat_points)):
                for j in range(i+1, len(cat_points)):
                    dists.append(np.linalg.norm(cat_points[i] - cat_points[j]))
            within_distances.append(np.mean(dists))
    
    # Calculate between-category distances
    between_distances = []
    for cat_id1 in range(len(unique_categories)):
        for cat_id2 in range(cat_id1+1, len(unique_categories)):
            cat1_points = embeddings_2d[category_ids == cat_id1]
            cat2_points = embeddings_2d[category_ids == cat_id2]
            if len(cat1_points) > 0 and len(cat2_points) > 0:
                # Calculate average distance between categories
                dists = []
                for p1 in cat1_points:
                    for p2 in cat2_points:
                        dists.append(np.linalg.norm(p1 - p2))
                between_distances.append(np.mean(dists))
    
    avg_within = np.mean(within_distances)
    avg_between = np.mean(between_distances)
    separation_ratio = avg_between / avg_within if avg_within > 0 else float('inf')
    
    return {
        "avg_within_category_distance": avg_within,
        "avg_between_category_distance": avg_between,
        "separation_ratio": separation_ratio
    }

def compare_results(pca_2d, tsne_2d, umap_2d, words, categories):
    """Compare results from different dimensionality reduction techniques"""
    pca_metrics = calculate_cluster_metrics(pca_2d, categories)
    tsne_metrics = calculate_cluster_metrics(tsne_2d, categories)
    umap_metrics = calculate_cluster_metrics(umap_2d, categories)
    
    results = pd.DataFrame({
        'PCA': [pca_metrics['avg_within_category_distance'], 
                pca_metrics['avg_between_category_distance'], 
                pca_metrics['separation_ratio']],
        't-SNE': [tsne_metrics['avg_within_category_distance'], 
                 tsne_metrics['avg_between_category_distance'], 
                 tsne_metrics['separation_ratio']],
        'UMAP': [umap_metrics['avg_within_category_distance'], 
                umap_metrics['avg_between_category_distance'], 
                umap_metrics['separation_ratio']]
    }, index=['Avg Within-Category Distance', 'Avg Between-Category Distance', 'Separation Ratio'])
    
    print("\nComparison of Dimensionality Reduction Techniques:")
    print(results)
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    results.iloc[2].plot(kind='bar', color=['blue', 'green', 'orange'])
    plt.title('Separation Ratio Comparison (Higher is Better)')
    plt.ylabel('Separation Ratio')
    plt.tight_layout()
    plt.savefig("separation_ratio_comparison.png")

if __name__ == "__main__":
    compare_visualization_techniques()
