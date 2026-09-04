"""D-GRanDe experiment runner.

Single entry point for the experiment battery. All global configuration lives
in this file:

* :class:`ExperimentConfig` holds the fixed pipeline hyperparameters.
* The CONTROL PANEL block selects which combinations of dataset, feature
  extractor, graph type, correlation, GCN type, sigma and number of folds are
  swept. Edit those constants to reproduce a given table from the paper.

Run with ``python main.py`` (with the working directory set so that the
``results/`` folder can be created).
"""
import csv
import itertools
import os
import random
import traceback
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from data_loader import get_dataset_paths, load_data
from graph_builder import (compute_coef_for_threshold, compute_correlation_matrix,
                           find_threshold, jaccardK, jaccardK_median, jaccardKMax,
                           rbo_matrix, run_ball_tree)
from gcn_model import GCNClassifier
from utils import compute_mean_std, create_folds


# ==========================================================
#  GLOBAL CONFIGURATION
# ==========================================================
@dataclass
class ExperimentConfig:
    """Fixed hyperparameters and general pipeline settings.

    Attributes:
        L: Fixed neighbor-list size used to build the BallTree ranked lists.
        k_graph: Number of neighbors considered when building the graph.
            Overwritten per run (200 for ``'dgcg'``, 40 otherwise).
        top_k: Fixed Top-K neighbors used by the correlation functions.
        learning_rate: Adam learning rate. Overwritten per run (0.01 for
            ``'cub200'``, 0.001 otherwise).
        epochs: Number of training epochs per fold.
        n_repetitions: Number of times the full fold evaluation is repeated.
    """
    L: int = 200
    k_graph: int = 200
    top_k: int = 40
    learning_rate: float = 0.001
    epochs: int = 200
    n_repetitions: int = 5


# ==========================================================
#  CONTROL PANEL — experiment battery selection
# ==========================================================
DATASETS = ['flowers']                 # e.g. ['flowers', 'cub200', 'pets', 'corel']
FEATURE_EXTRACTORS = ['resnet']        # e.g. ['resnet', 'senet', 'vit']

GRAPH_TYPES = ['dgcg']                 # e.g. ['knn', 'rec', 'dgcg']
DGCG_CORRELATIONS = ['rbo']            # used only when graph_type == 'dgcg'

GCN_TYPES = ['appnp_grande']           # e.g. ['sgc_degree', 'sgc_grande', 'appnp_degree', 'appnp_grande']
SIGMA_VALUES = [0.2]                   # e.g. [0.1, 0.5, 1.0]; used only for '*_grande' GCNs

N_FOLDS_OPTIONS = [10]                 # e.g. [3, 5, 10]

# Threshold selection for the DGCG graph.
USE_AUTOMATIC_THRESHOLD = True
MANUAL_THRESHOLDS = [0.4]              # used only when USE_AUTOMATIC_THRESHOLD is False

# Available correlation functions for the DGCG graph.
CORRELATION_FUNCTIONS = {
    'rbo': rbo_matrix,
    'jaccard_max': jaccardKMax,
    'jaccard_median': jaccardK_median,
    'jaccardk': jaccardK,
}


# ==========================================================
#  EXPERIMENT BATTERY ORCHESTRATION
# ==========================================================
def main():
    """Run the full experiment battery and append every result to a CSV."""
    config = ExperimentConfig()

    grid = generate_experiment_grid(
        DATASETS, FEATURE_EXTRACTORS, GRAPH_TYPES, DGCG_CORRELATIONS,
        N_FOLDS_OPTIONS, GCN_TYPES, SIGMA_VALUES,
    )

    print(f"\nTotal experiments to run: {len(grid)}\n")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f'results/results_batch_{timestamp}.csv'
    os.makedirs('results', exist_ok=True)

    for manual_threshold in MANUAL_THRESHOLDS:
        for i, params in enumerate(grid, start=1):
            print(f"\n[{i}/{len(grid)}] Running combination: {params}")
            try:
                result = run_single_experiment(
                    params, config,
                    use_automatic_threshold=USE_AUTOMATIC_THRESHOLD,
                    manual_threshold=manual_threshold,
                )
                append_result_to_csv(result, csv_path)
                print(f"    Accuracy: {result['accuracy_mean']} $\\pm$ {result['accuracy_std_per_repetition']}")

            except Exception as e:
                # A single failing experiment must not bring down the whole battery.
                print(f"    ERROR in experiment {params}: {e}")
                traceback.print_exc()
                error_result = dict(params)
                error_result['error'] = str(e)
                append_result_to_csv(error_result, csv_path.replace('.csv', '_errors.csv'))
                continue

        print("\n" + "=" * 40)
        print(f" Battery finished. Results in: {csv_path}")
        print("=" * 40)


# ==========================================================
#  1. EXPERIMENT GRID GENERATION
# ==========================================================
def generate_experiment_grid(datasets, feature_extractors, graph_types,
                             dgcg_correlations, n_folds_options=(10,),
                             gcn_types=('sgc_grande', 'appnp_grande'),
                             sigma_values=(0.2,)):
    """Generate the list of parameter combinations to evaluate.

    Conditional rules avoid invalid/redundant combinations:
      * ``correlation`` only varies when ``graph_type == 'dgcg'``. For ``'knn'``
        or ``'rec'`` it is fixed to ``None`` so the same experiment is not run
        several times.
      * ``sigma`` only varies for ``'*_grande'`` GCNs; otherwise it is ``None``.

    Args:
        datasets: Dataset names to sweep.
        feature_extractors: Feature extractor keys to sweep.
        graph_types: Graph construction methods to sweep.
        dgcg_correlations: Correlation functions to sweep (DGCG only).
        n_folds_options: Fold counts to sweep.
        gcn_types: GCN variants to sweep.
        sigma_values: Sigma values to sweep (``*_grande`` GCNs only).

    Returns:
        List of parameter dicts, one per experiment combination.
    """
    grid = []
    for dataset, fe, graph_type in itertools.product(datasets, feature_extractors, graph_types):
        correlations_to_use = dgcg_correlations if graph_type == 'dgcg' else [None]
        for correlation in correlations_to_use:
            for gcn_type in gcn_types:
                sigmas = sigma_values if 'grande' in gcn_type else [None]
                for sigma in sigmas:
                    for n_folds in n_folds_options:
                        grid.append({
                            'dataset': dataset,
                            'feature_extractor': fe,
                            'graph_type': graph_type,
                            'dgcg_correlation': correlation,
                            'gcn_type': gcn_type,
                            'sigma': sigma,
                            'n_folds': n_folds,
                        })
    return grid


# ==========================================================
#  2. SINGLE EXPERIMENT EXECUTION
# ==========================================================
def run_single_experiment(params, config, use_automatic_threshold=True, manual_threshold=0.4):
    """Run one experiment combination end to end.

    Args:
        params: Dict with keys ``'dataset'``, ``'feature_extractor'``,
            ``'graph_type'``, ``'dgcg_correlation'``, ``'gcn_type'``,
            ``'sigma'`` and ``'n_folds'``.
        config: :class:`ExperimentConfig` instance (mutated in place with the
            per-dataset hyperparameter overrides).
        use_automatic_threshold: If True, search the DGCG threshold to hit the
            target density; otherwise use ``manual_threshold``.
        manual_threshold: Threshold used when ``use_automatic_threshold`` is
            False.

    Returns:
        Dict with the aggregated metrics for this combination.
    """
    dataset = params['dataset']
    feature_extractor = params['feature_extractor']
    graph_type = params['graph_type']
    dgcg_correlation = params['dgcg_correlation']
    gcn_type = params['gcn_type']
    sigma = params['sigma']
    n_folds = params['n_folds']

    # Dataset-specific hyperparameter overrides.
    config.learning_rate = 0.01 if dataset == 'cub200' else 0.001
    config.k_graph = 200 if graph_type == 'dgcg' else 40

    # --- Data loading ---
    paths = get_dataset_paths(dataset)
    print(f"--> Loading dataset '{dataset}' with '{feature_extractor}' features...")
    features, labels = load_data(dataset, feature_extractor, paths)

    # --- Ranked lists ---
    ranked_lists = run_ball_tree(features, k=config.L)

    # --- Topology (DGCG only) ---
    correlation_matrix = 0
    threshold = 0
    density = None

    if graph_type == 'dgcg':
        if dgcg_correlation not in CORRELATION_FUNCTIONS:
            raise ValueError(f"Correlation function '{dgcg_correlation}' not recognized.")

        correlation_func = CORRELATION_FUNCTIONS[dgcg_correlation]
        correlation_matrix = compute_correlation_matrix(
            ranked_lists, top_k=config.top_k, L=config.L,
            correlation_func=correlation_func,
        )

        if use_automatic_threshold:
            threshold, density = find_threshold(
                correlation_matrix=correlation_matrix,
                ranked_lists=ranked_lists,
                k_graph=config.k_graph,
            )
        else:
            threshold = manual_threshold
            density = compute_coef_for_threshold(
                correlation_matrix, ranked_lists, threshold, config.k_graph,
            )

    # --- Seeding and training ---
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"\n--- Experiment: {dataset} | {feature_extractor} | {graph_type.upper()}"
          f"{'(' + dgcg_correlation + ')' if dgcg_correlation else ''} ---")

    clf = GCNClassifier(gcn_type=gcn_type, rks=ranked_lists, pN=len(features), config=config, sigma=sigma)
    folds = create_folds(features, labels, n_folds=n_folds)

    accuracy_per_repetition = []
    accuracy_all_folds = []
    f1_per_repetition = []
    precision_per_repetition = []
    recall_per_repetition = []

    for _ in range(config.n_repetitions):
        fold_accuracy = []
        fold_f1 = []
        fold_precision = []
        fold_recall = []

        for test_idx, train_idx in tqdm(folds, desc="Evaluating folds", leave=False):
            clf.prepare(
                test_index=test_idx, train_index=train_idx, features=features,
                labels=labels, graph_type=graph_type, matrix=correlation_matrix,
                threshold=threshold,
            )
            _, predictions = clf.train_and_predict()

            test_labels = [labels[i] for i in test_idx]
            acc = sum(1 for i, p in enumerate(predictions) if test_labels[i] == p) / len(predictions)
            f1 = f1_score(test_labels, predictions, average='macro')
            precision = precision_score(test_labels, predictions, average='macro', zero_division=0)
            recall = recall_score(test_labels, predictions, average='macro')

            fold_accuracy.append(acc)
            fold_f1.append(f1)
            fold_precision.append(precision)
            fold_recall.append(recall)
            accuracy_all_folds.append(acc)

        accuracy_per_repetition.append(np.mean(fold_accuracy))
        f1_per_repetition.append(np.mean(fold_f1))
        precision_per_repetition.append(np.mean(fold_precision))
        recall_per_repetition.append(np.mean(fold_recall))

    # Primary metric: mean/std computed over the per-repetition means, so the
    # standard deviation reflects the variability *between repetitions*.
    accuracy_mean, accuracy_std_per_repetition = compute_mean_std(accuracy_per_repetition)
    # Alternative: mean/std computed over every individual fold pooled together,
    # so the standard deviation reflects the variability *between folds*.
    accuracy_mean_per_fold, accuracy_std_per_fold = compute_mean_std(accuracy_all_folds)

    del clf
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Consolidated result for this experiment ---
    result = {
        'dataset': dataset,
        'feature_extractor': feature_extractor,
        'graph_type': graph_type,
        'gcn_type': gcn_type,
        'sigma': sigma,
        'n_folds': n_folds,
        'dgcg_correlation': dgcg_correlation if graph_type == 'dgcg' else '-',
        'threshold': round(threshold, 4) if graph_type == 'dgcg' else '-',
        'density': round(density, 4) if density is not None else '-',
        'f1_mean': round(np.mean(f1_per_repetition) * 100, 4),
        'f1_std': round(np.std(f1_per_repetition) * 100, 4),
        'precision_mean': round(np.mean(precision_per_repetition) * 100, 4),
        'precision_std': round(np.std(precision_per_repetition) * 100, 4),
        'recall_mean': round(np.mean(recall_per_repetition) * 100, 4),
        'recall_std': round(np.std(recall_per_repetition) * 100, 4),
        # Accuracy aggregated over per-repetition means (std between repetitions).
        'accuracy_mean': round(accuracy_mean * 100, 4),
        'accuracy_std_per_repetition': round(accuracy_std_per_repetition * 100, 4),
        # Accuracy aggregated over all individual folds (std between folds).
        'accuracy_mean_per_fold': round(accuracy_mean_per_fold * 100, 4),
        'accuracy_std_per_fold': round(accuracy_std_per_fold * 100, 4),
        'accuracy_latex': f"{round(accuracy_mean * 100, 2)} $\\pm$ {round(accuracy_std_per_repetition * 100, 2)}",
        'accuracy_all_folds': accuracy_all_folds,
    }
    return result


def append_result_to_csv(result, csv_path):
    """Append a result dict as a row to ``csv_path``, writing a header if new.

    Args:
        result: Result dict (its keys become the CSV columns).
        csv_path: Destination CSV path.
    """
    file_exists = os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


if __name__ == '__main__':
    main()
