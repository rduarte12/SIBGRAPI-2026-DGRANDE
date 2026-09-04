import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.nn import SGConv, APPNP

from grande_models import GrandeAPPNP, GrandeSGConv


class GrandeAPPNPNet(torch.nn.Module):
    """Two-layer MLP followed by GRaNDe-weighted APPNP propagation.

    Args:
        num_features: Number of input features.
        num_classes: Number of output classes.
        neurons: Hidden layer width.
        sigma: Bandwidth of the Gaussian kernel used by GRaNDe.
        K: Number of APPNP propagation steps.
        alpha: APPNP teleport probability.
    """

    def __init__(self, num_features, num_classes, neurons=256, sigma=0.2, K=10, alpha=0.1):
        super(GrandeAPPNPNet, self).__init__()
        hidden = neurons
        self.K = K
        self.alpha = alpha
        self.lin1 = Linear(num_features, hidden)
        self.lin2 = Linear(hidden, num_classes)
        self.prop1 = GrandeAPPNP(self.K, self.alpha, sigma=sigma)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        dropout = 0.5
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)


class GrandeSGC(torch.nn.Module):
    """Single-layer SGC using GRaNDe-based normalization.

    Args:
        num_features: Number of input features.
        num_classes: Number of output classes.
        sigma: Bandwidth of the Gaussian kernel used by GRaNDe.
    """

    def __init__(self, num_features, num_classes, sigma=0.2):
        super(GrandeSGC, self).__init__()
        self.conv1 = GrandeSGConv(num_features, num_classes, K=2, cached=False, sigma=sigma)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)


class APPNPNet(torch.nn.Module):
    """Two-layer MLP followed by standard (degree-normalized) APPNP propagation.

    Args:
        num_features: Number of input features.
        num_classes: Number of output classes.
        neurons: Hidden layer width.
        K: Number of APPNP propagation steps.
        alpha: APPNP teleport probability.
    """

    def __init__(self, num_features, num_classes, neurons=256, K=10, alpha=0.1):
        super(APPNPNet, self).__init__()
        hidden = neurons
        self.K = K
        self.alpha = alpha
        self.lin1 = Linear(num_features, hidden)
        self.lin2 = Linear(hidden, num_classes)
        self.prop1 = APPNP(self.K, self.alpha, cached=True)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        dropout = 0.5
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)


class SGC(torch.nn.Module):
    """Single-layer standard Simplified Graph Convolution.

    Args:
        num_features: Number of input features.
        num_classes: Number of output classes.
    """

    def __init__(self, num_features, num_classes):
        super(SGC, self).__init__()
        self.conv1 = SGConv(num_features, num_classes, K=2, cached=True)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)


class GCNClassifier:
    """Trainer/orchestrator around the four supported GCN variants.

    Holds the ranked lists and hyperparameters, builds the transductive graph
    for a given fold and trains a model to predict the test nodes.

    Args:
        gcn_type: One of ``'sgc_degree'``, ``'sgc_grande'``, ``'appnp_degree'``,
            ``'appnp_grande'``.
        rks: Ranked lists ``(n_samples, L)`` used to build the graph.
        pN: Total number of nodes (samples).
        config: ExperimentConfig instance providing ``k_graph``,
            ``learning_rate`` and ``epochs``.
        sigma: Bandwidth of the Gaussian kernel (used by ``*_grande`` variants).
    """

    def __init__(self, gcn_type, rks, pN, config, sigma=0.2):
        self.pK = config.k_graph
        self.pN = pN
        self.rks = rks
        self.pLR = config.learning_rate
        self.pNNeurons = 256
        self.pNEpochs = config.epochs
        self.gcn_type = gcn_type
        self.sigma = sigma

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print("INFO [GCNClassifier]: Running on GPU.")
        else:
            print("WARNING [GCNClassifier]: GPU not found, running on CPU.")

    def prepare(self, test_index, train_index, features, labels, graph_type, matrix, threshold):
        """Set up masks, tensors and the graph for a single fold.

        Args:
            test_index: Indices of the test nodes.
            train_index: Indices of the train nodes.
            features: Feature matrix ``(n_samples, n_features)``.
            labels: Label vector ``(n_samples,)``.
            graph_type: One of ``'knn'``, ``'rec'``, ``'dgcg'``.
            matrix: Correlation matrix (used only by ``'dgcg'``).
            threshold: Correlation threshold (used only by ``'dgcg'``).
        """
        self.train_mask = [False] * self.pN
        self.val_mask = [False] * self.pN
        self.test_mask = [False] * self.pN

        for index in train_index:
            self.train_mask[index] = True
        for index in test_index:
            self.test_mask[index] = True

        self.train_mask = torch.tensor(self.train_mask)
        self.val_mask = torch.tensor(self.val_mask)
        self.test_mask = torch.tensor(self.test_mask)

        y = labels
        self.num_classes = max(y) + 1
        self.y = torch.tensor(y)

        self.x = torch.tensor(features)
        self.pNFeatures = len(features[0])

        self.create_graph(graph_type, matrix, threshold)

    def dgcg(self, k_graph, threshold, matrix):
        """Build the Density-Guided Correlation Graph (DGCG).

        Adds an edge ``(img1, img2)`` when ``img2`` is among the first
        ``k_graph`` neighbors of ``img1`` and their correlation exceeds
        ``threshold``.

        Args:
            k_graph: Number of candidate neighbors per node.
            threshold: Correlation threshold above which an edge is kept.
            matrix: Pairwise correlation matrix.
        """
        edge_index = []

        for img1 in range(len(self.rks)):
            for pos in range(k_graph):
                img2 = self.rks[img1][pos]

                if img1 == img2:
                    continue

                if matrix[img1][img2] > threshold:
                    edge_index.append([img1, img2])

        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    def rec(self, top_k):
        """Build the reciprocal-neighbor graph.

        Args:
            top_k: Neighborhood size used to determine reciprocity.
        """
        ref_list = [[] for _ in range(self.pN)]
        for img1 in range(len(self.rks)):
            for pos in range(top_k):
                img2 = self.rks[img1][pos]
                ref_list[img2].append(img1)
        edge_index = []
        for img1 in range(len(self.rks)):
            for pos in range(self.pK):
                img2 = self.rks[img1][pos]
                if img2 in ref_list[img1]:
                    edge_index.append([img1, img2])
        edge_index = torch.tensor(edge_index)
        self.edge_index = edge_index.t().contiguous()

    def knn(self, top_k):
        """Build the k-nearest-neighbor graph.

        Args:
            top_k: Number of neighbors connected per node.
        """
        edge_index = []
        for img1 in range(len(self.rks)):
            for pos in range(top_k):
                img2 = self.rks[img1][pos]
                edge_index.append([img1, img2])
        edge_index = torch.tensor(edge_index)
        self.edge_index = edge_index.t().contiguous()

    def create_graph(self, graph_type, matrix, threshold):
        """Dispatch to the appropriate graph-construction routine.

        Args:
            graph_type: One of ``'knn'``, ``'rec'``, ``'dgcg'``.
            matrix: Correlation matrix (used only by ``'dgcg'``).
            threshold: Correlation threshold (used only by ``'dgcg'``).

        Raises:
            ValueError: If ``graph_type`` is not supported.
        """
        if graph_type == 'knn':
            self.knn(self.pK)
        elif graph_type == 'rec':
            self.rec(self.pK)
        elif graph_type == 'dgcg':
            self.dgcg(self.pK, threshold, matrix)
        else:
            raise ValueError(f"Graph type '{graph_type}' not supported.")

    def build_model(self):
        """Instantiate the model selected by ``self.gcn_type`` on the device.

        Raises:
            ValueError: If ``self.gcn_type`` is not recognized.
        """
        if self.gcn_type == 'sgc_degree':
            return SGC(self.pNFeatures, self.num_classes).to(self.device)
        elif self.gcn_type == 'sgc_grande':
            return GrandeSGC(self.pNFeatures, self.num_classes, sigma=self.sigma).to(self.device)
        elif self.gcn_type == 'appnp_degree':
            return APPNPNet(self.pNFeatures, self.num_classes, neurons=self.pNNeurons).to(self.device)
        elif self.gcn_type == 'appnp_grande':
            return GrandeAPPNPNet(self.pNFeatures, self.num_classes, neurons=self.pNNeurons, sigma=self.sigma).to(self.device)
        else:
            raise ValueError(f"GCN type '{self.gcn_type}' not recognized.")

    def train_and_predict(self):
        """Train the model on the current fold and predict the test nodes.

        Returns:
            Tuple ``(embeddings, predictions)`` where ``embeddings`` are the
            (CPU) log-softmax outputs for all nodes and ``predictions`` is the
            list of predicted classes for the test nodes.
        """
        data = Data(x=self.x.float(), y=self.y,
                    test_mask=self.test_mask, train_mask=self.train_mask,
                    val_mask=self.val_mask, edge_index=self.edge_index).to(self.device)

        model = self.build_model()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.pLR, weight_decay=5e-4)

        model.train()
        for _ in range(self.pNEpochs):
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            out_final = model(data)
            pred_all = out_final.argmax(dim=1)
            pred = torch.masked_select(pred_all, data.test_mask).cpu()
            embeddings = out_final.cpu()

        del data, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return embeddings, pred.tolist()
