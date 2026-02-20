import numpy as np
import torch
import torch.nn.functional as F
from ml_library import get_hooked_features

@torch.no_grad()
def evaluate_few_shot(model, hook_manager, dev, valid_ds, n_way=5, k_shot=5, n_episodes=500):
    """
    Evaluates few-shot performance by capturing raw features via forward hooks.
    """
    model.eval()

    # 2. Organize data by class for sampling
    class_data = {}
    for img, label in valid_ds:
        # Check if the label is a Tensor; if so, use .item(), otherwise use it directly
        target = label.item() if hasattr(label, 'item') else label

        if target not in class_data:
            class_data[target] = []
        class_data[target].append(img)

    available_classes = list(class_data.keys())
    accuracies = []

    # 3. Episode Loop
    for _ in range(n_episodes):
        episode_classes = np.random.choice(available_classes, n_way, replace=False)
        prototypes = []
        queries = []
        query_labels = []

        for i, cls in enumerate(episode_classes):
            imgs = class_data[cls]
            # Sample k for support and 5 for query
            indices = np.random.choice(len(imgs), k_shot + 5, replace=False)

            support_imgs = torch.stack([imgs[idx] for idx in indices[:k_shot]]).to(dev)
            query_imgs = torch.stack([imgs[idx] for idx in indices[k_shot:]]).to(dev)

            # Use our hook helper to get raw expert signals
            support_features = get_hooked_features(model, hook_manager, support_imgs)
            prototypes.append(support_features.mean(0))

            queries.append(get_hooked_features(model, hook_manager, query_imgs))
            query_labels.extend([i] * len(query_imgs))

        prototypes = torch.stack(prototypes)
        queries = torch.cat(queries)
        query_labels = torch.tensor(query_labels).to(dev)

        # 4. Normalization as proxy for cosine distance
        # This acts on the un-squashed, high-magnitude HT features
        support_norm = F.normalize(prototypes, p=2, dim=-1)
        query_norm = F.normalize(queries, p=2, dim=-1)

        dists = torch.cdist(query_norm, support_norm)
        preds = dists.argmin(1)

        acc = (preds == query_labels).float().mean().item()
        accuracies.append(acc)

    # Return the raw list instead of the mean
    return accuracies
