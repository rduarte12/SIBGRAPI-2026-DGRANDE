import numpy as np
from sklearn.neighbors import BallTree


def run_ball_tree(features, k):
    """Build a BallTree over the features and return the ranked neighbor lists.

    Args:
        features: Feature matrix ``(n_samples, n_features)`` as a numpy array.
        k: Number of nearest neighbors to retrieve per sample.

    Returns:
        Array ``(n_samples, k)`` with the indices of the ``k`` nearest
        neighbors of each sample (the ranked list).

    Raises:
        ValueError: If ``features`` is not a 2D numpy array.
    """
    if not isinstance(features, np.ndarray):
        raise ValueError('Features must be a numpy array.')
    if features.ndim != 2:
        raise ValueError('Features must be a 2D array (n_samples, n_features).')

    tree = BallTree(features, leaf_size=200)
    _, ranked_lists = tree.query(features, k=k)
    return ranked_lists


def rbo_matrix(rks1, rks2, top_k):
    """Rank-Biased Overlap (RBO) similarity between two ranked lists.

    Args:
        rks1: First ranked list (sequence of neighbor indices).
        rks2: Second ranked list.
        top_k: Number of top positions to consider.

    Returns:
        RBO score in ``[0, 1]``.
    """
    stored = set()
    r = 0.9
    acum_inter = 0
    score = 0
    img1_leftover = set()
    img2_leftover = set()

    for k in range(top_k):
        img1_elm = rks1[k]
        img2_elm = rks2[k]

        if img1_elm not in stored and img1_elm == img2_elm:
            acum_inter += 1
            stored.add(img1_elm)
        else:
            if img1_elm not in stored:
                if img1_elm in img2_leftover:
                    acum_inter += 1
                    stored.add(img1_elm)
                    img2_leftover.remove(img1_elm)
                else:
                    img1_leftover.add(img1_elm)
            if img2_elm not in stored:
                if img2_elm in img1_leftover:
                    acum_inter += 1
                    stored.add(img2_elm)
                    img1_leftover.remove(img2_elm)
                else:
                    img2_leftover.add(img2_elm)

        score += (r ** k) * (acum_inter / (k + 1))

    normalized_score = (1 - r) * score
    return normalized_score


def jaccardK_median(rks1, rks2, top_k):
    """Median of the intersection-over-union computed at each depth ``k``.

    Args:
        rks1: First ranked list.
        rks2: Second ranked list.
        top_k: Number of top positions to consider.

    Returns:
        Median Jaccard@k score.
    """
    scores = []
    stored = set()
    stored_img1 = set()
    stored_img2 = set()
    acum_inter = 0
    img1_leftover = set()
    img2_leftover = set()

    for k in range(top_k):
        img1_elm = rks1[k]
        img2_elm = rks2[k]

        if img1_elm not in stored and img1_elm == img2_elm:
            acum_inter += 1
            stored.add(img1_elm)
            stored_img1.add(img1_elm)
            stored_img2.add(img2_elm)
        else:
            if img1_elm not in stored:
                if img1_elm in img2_leftover:
                    acum_inter += 1
                    stored.add(img1_elm)
                    img2_leftover.remove(img1_elm)
                    stored_img1.add(img1_elm)
                else:
                    img1_leftover.add(img1_elm)
                    stored_img1.add(img1_elm)
            if img2_elm not in stored:
                if img2_elm in img1_leftover:
                    acum_inter += 1
                    stored.add(img2_elm)
                    img1_leftover.remove(img2_elm)
                    stored_img2.add(img2_elm)
                else:
                    img2_leftover.add(img2_elm)
                    stored_img2.add(img2_elm)

        denominator = len(stored_img1) + len(stored_img2) - acum_inter
        if denominator > 0:
            score = acum_inter / denominator
            scores.append(score)

    return np.median(scores)


def jaccardK(rks1, rks2, top_k):
    """Average of the intersection-over-union computed at each depth ``k``.

    Args:
        rks1: First ranked list.
        rks2: Second ranked list.
        top_k: Number of top positions to consider.

    Returns:
        Mean Jaccard@k score.
    """
    score = 0
    x_leftover = set()
    y_leftover = set()
    stored = set()
    stored_x = set()
    stored_y = set()
    cur_inter = 0

    for i in range(top_k):
        x_elm = rks1[i]
        y_elm = rks2[i]
        if x_elm not in stored and x_elm == y_elm:
            cur_inter += 1
            stored.add(x_elm)
            stored_x.add(x_elm)
            stored_y.add(y_elm)
        else:
            if x_elm not in stored:
                if x_elm in y_leftover:
                    cur_inter += 1
                    stored.add(x_elm)
                    stored_x.add(x_elm)
                    y_leftover.remove(x_elm)
                else:
                    x_leftover.add(x_elm)
                    stored_x.add(x_elm)
            if y_elm not in stored:
                if y_elm in x_leftover:
                    cur_inter += 1
                    stored.add(y_elm)
                    stored_y.add(y_elm)
                    x_leftover.remove(y_elm)
                else:
                    y_leftover.add(y_elm)
                    stored_y.add(y_elm)

        score += cur_inter / (len(stored_x) + len(stored_y) - cur_inter)

    return score / top_k


def jaccardKMax(rks1, rks2, top_k):
    """Maximum intersection-over-union observed across depths ``k``.

    Args:
        rks1: First ranked list.
        rks2: Second ranked list.
        top_k: Number of top positions to consider.

    Returns:
        Maximum Jaccard@k score.
    """
    stored = set()
    stored_img1 = set()
    stored_img2 = set()
    acum_inter = 0
    max_score = 0
    score = 0
    img1_leftover = set()
    img2_leftover = set()

    for k in range(top_k):
        img1_elm = rks1[k]
        img2_elm = rks2[k]

        if img1_elm not in stored and img1_elm == img2_elm:
            acum_inter += 1
            stored.add(img1_elm)
            stored_img1.add(img1_elm)
            stored_img2.add(img2_elm)
        else:
            if img1_elm not in stored:
                if img1_elm in img2_leftover:
                    acum_inter += 1
                    stored.add(img1_elm)
                    img2_leftover.remove(img1_elm)
                    stored_img1.add(img1_elm)
                else:
                    img1_leftover.add(img1_elm)
                    stored_img1.add(img1_elm)
            if img2_elm not in stored:
                if img2_elm in img1_leftover:
                    acum_inter += 1
                    stored.add(img2_elm)
                    img1_leftover.remove(img2_elm)
                    stored_img2.add(img2_elm)
                else:
                    img2_leftover.add(img2_elm)
                    stored_img2.add(img2_elm)

        denominator = len(stored_img1) + len(stored_img2) - acum_inter
        if denominator > 0:
            score = acum_inter / denominator
            if score > max_score:
                max_score = score

    return max_score


def compute_correlation_matrix(ranked_lists, top_k, L, correlation_func=rbo_matrix):
    """Build the pairwise correlation matrix over the ranked lists.

    For each sample ``i``, the correlation is computed against its first ``L``
    neighbors only (the matrix stays sparse in practice).

    Args:
        ranked_lists: Array ``(n_samples, L)`` of neighbor indices.
        top_k: Number of top positions passed to ``correlation_func``.
        L: Number of neighbors of each sample to correlate against.
        correlation_func: One of the correlation functions in this module.

    Returns:
        Dense correlation matrix ``(n_samples, n_samples)``.
    """
    n_samples = len(ranked_lists)
    correlation_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(L):
            neighbor_j = ranked_lists[i][j]
            score = correlation_func(ranked_lists[i], ranked_lists[neighbor_j], top_k)
            correlation_matrix[i, neighbor_j] = score

    return correlation_matrix


def compute_coef_for_threshold(correlation_matrix, ranked_lists, threshold, k_graph):
    """Compute the graph density that a given correlation threshold yields.

    Args:
        correlation_matrix: Pairwise correlation matrix.
        ranked_lists: Array ``(n_samples, L)`` of neighbor indices.
        threshold: Correlation threshold above which an edge is kept.
        k_graph: Number of candidate neighbors per node to inspect.

    Returns:
        Edge density: kept edges over the maximum possible number of edges.
    """
    n_samples = correlation_matrix.shape[0]
    num_edges = 0

    for i in range(n_samples):
        for j in range(k_graph):
            neighbor_idx = ranked_lists[i][j]
            if correlation_matrix[i, neighbor_idx] > threshold:
                num_edges += 1

    total_possible_edges = n_samples * 200

    return num_edges / total_possible_edges


def find_threshold(correlation_matrix, ranked_lists, k_graph, initial_threshold=0.5,
                   target_density=(0.03, 0.04), max_iter=10):
    """Binary-search the correlation threshold that hits a target graph density.

    Args:
        correlation_matrix: Pairwise correlation matrix.
        ranked_lists: Array ``(n_samples, L)`` of neighbor indices.
        k_graph: Number of candidate neighbors per node used to count edges.
        initial_threshold: Starting threshold for the search.
        target_density: ``(low, high)`` inclusive target density interval.
        max_iter: Maximum number of bisection iterations.

    Returns:
        Tuple ``(threshold, density)`` for the best threshold found (either
        within the target interval or the closest one after ``max_iter``).
    """
    print("\n--- Starting automatic search for the ideal threshold ---")

    low, high = 0.0, 1.0
    best_threshold = initial_threshold

    for i in range(1, max_iter + 1):
        coef = compute_coef_for_threshold(correlation_matrix, ranked_lists, best_threshold, k_graph)
        print(f'Iteration {i}: threshold = {best_threshold:.4f}, density = {coef:.4f}')

        if target_density[0] <= coef <= target_density[1]:
            print(f"\nIdeal threshold found within the target interval in {i} iterations.")
            return best_threshold, coef

        if coef < target_density[0]:
            high = best_threshold
            best_threshold = (low + high) / 2
        else:
            low = best_threshold
            best_threshold = (low + high) / 2

    print("\nSearch finished. Could not converge to the exact target interval.")
    print(f"Returning the closest threshold found: {best_threshold:.4f}")
    return best_threshold, coef
