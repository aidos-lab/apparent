"Plotting functions for Physician Referral Networks"


# Curvature distributions
# Embeddings
# Individual Graphs


def plot_phate_embedding(
    distance_matrix,
    n_components=2,
    knn=10,
    decay=40,
    njobs=1,
    year=2014,
    feature="OR_0",
):
    """Plot and Clusterusing PHATE."""
    phate_operator = phate.PHATE(
        n_components=n_components,
        knn=knn,
        decay=decay,
        knn_dist="precomputed",
        n_jobs=njobs,
    )
    phate_embedding = phate_operator.fit_transform(distance_matrix)
    clusters = phate.cluster.kmeans(phate_operator, n_clusters="auto")

    fig = phate.plot.scatter2d(
        phate_operator,
        c=clusters,
        title=f"{year} Physician Referral Networks: Measured by {feature}",
    )
    plt.xlabel("PHATE 1")
    plt.ylabel("PHATE 2")

    return fig, phate_embedding


def plot_dendrogram(model, ids):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    dendrogram(
        linkage_matrix,
        labels=ids,
    )
    # Plot dendrogram
    plt.title("2014 Sample Physician Referral Networks")
    plt.xlabel("hsanum")
    plt.ylabel("Curvature Filtrations Distance")
    plt.show()
