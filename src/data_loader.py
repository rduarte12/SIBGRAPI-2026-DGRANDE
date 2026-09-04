import os

import numpy as np

# Project root: the D-GRanDe directory, derived from this file's location
# (src/data_loader.py -> parent of src/). This keeps the data paths valid
# regardless of the working directory the pipeline is launched from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_dataset_paths(dataset_name):
    """Return the file paths associated with a given dataset.

    Uses a single centralized mapping of file names per dataset.

    Args:
        dataset_name: One of ``'flowers'``, ``'cub200'``, ``'pets'``,
            ``'corel'``.

    Returns:
        Dict with the resolved paths (``data_dir``, ``list_path``,
        ``class_path``, ``resnet_path``, ``senet_path``, ``vit_path``,
        ``main_path``, ``bin_path``). Entries whose file name is undefined are
        set to ``None``.

    Raises:
        ValueError: If ``dataset_name`` is not supported.
    """
    data_dir = os.path.join(PROJECT_ROOT, "data", "Features-Labels-Lists")
    bin_path = os.path.join(PROJECT_ROOT, "UDLF", "bin")

    # Centralized mapping of path configurations for each dataset.
    dataset_configs = {
        'flowers': {
            'list_name': "listFlowers.txt",
            'class_name': "labels.txt",
            'resnet_name': "cnn-last_linear-resnet152.npz",
            'senet_name': "cnn-last_linear-senet154.npz",
            'vit_name': "features_vit-b16_flowers.npy",
        },
        'cub200': {
            'list_name': "listCub.txt",
            'class_name': "labels_cub200.npy",
            'resnet_name': "cub200_resnet152.npy",
            'senet_name': "cub200_senet.npy",
            'vit_name': "features_vit_cub200.npy",
        },
        'pets': {
            'list_name': "pets_lists.txt",
            'class_name': "pets_labels.txt",
            'resnet_name': "pets_resnet.npy",
            'senet_name': "pets_senet.npy",
            'vit_name': "pets_vit16.npy",
        },
        'corel': {
            'list_name': "corel5k_lists.txt",
            'class_name': "corel5k_labels.txt",
            'resnet_name': "cnn-last_linear-resnet152_corel.npz",
            'senet_name': None,  # Not supported in the original pipeline.
            'vit_name': "features_vit-b16_corel5k.npy",
        },
    }

    if dataset_name not in dataset_configs:
        raise ValueError(f"Dataset '{dataset_name}' not supported.")
    config = dataset_configs[dataset_name]

    paths = {
        'data_dir': data_dir,
        'list_path': os.path.join(data_dir, config['list_name']) if config.get('list_name') else None,
        'class_path': os.path.join(data_dir, config['class_name']) if config.get('class_name') else None,
        'resnet_path': os.path.join(data_dir, config['resnet_name']) if config.get('resnet_name') else None,
        'senet_path': os.path.join(data_dir, config['senet_name']) if config.get('senet_name') else None,
        'vit_path': os.path.join(data_dir, config['vit_name']) if config.get('vit_name') else None,
        'main_path': PROJECT_ROOT,
        'bin_path': bin_path,
    }

    return paths


def load_data(dataset_name, feat_extractor, paths):
    """Load the features and labels of the selected dataset.

    Args:
        dataset_name: Dataset identifier (see :func:`get_dataset_paths`).
        feat_extractor: Feature extractor key (``'resnet'``, ``'senet'`` or
            ``'vit'``).
        paths: Path dictionary returned by :func:`get_dataset_paths`.

    Returns:
        Tuple ``(features, labels)`` as numpy arrays.

    Raises:
        ValueError: If the extractor or dataset is unsupported, or if the
            feature file format is not recognized.
    """
    # 1. Unified feature loading.
    extractor_key = f"{feat_extractor}_path"
    if extractor_key not in paths or not paths[extractor_key]:
        raise ValueError(f"Feature extractor '{feat_extractor}' not supported for '{dataset_name}'.")

    feat_path = paths[extractor_key]

    # Detect the file format dynamically and load the features accordingly.
    if feat_path.endswith('.npz'):
        features = np.load(feat_path)['features']
    elif feat_path.endswith('.npy'):
        features = np.load(feat_path)
    else:
        raise ValueError(f"Unrecognized feature file format at: {feat_path}")

    # 2. Label loading (conditioned on the physical layout of each dataset).
    class_path = paths['class_path']
    list_path = paths['list_path']

    if dataset_name == 'flowers':
        with open(list_path, 'r') as file:
            dataset_elements = [line.strip() for line in file.readlines()]
        class_size = 80
        labels = [i // class_size for i in range(len(dataset_elements))]

    elif dataset_name == 'cub200':
        labels = np.load(class_path)

    elif dataset_name in ['pets', 'corel']:
        labels = []
        with open(class_path, 'r') as file:
            for line in file:
                _, cls = line.strip().split(":")
                labels.append(int(cls))
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported for label formatting.")

    return np.array(features), np.array(labels)
