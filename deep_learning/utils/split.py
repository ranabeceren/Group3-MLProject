import numpy as np

def train_val_test_split(images,
                         masks,
                         train,
                         val,
                         seed=42):
    
    np.random.seed(seed)

    indices = np.arange(len(images))
    np.random.shuffle(indices)

    num_train = int(len(indices) * train)
    num_val = int(len(indices) * val)

    train_idx = indices[:num_train]
    val_idx = indices[num_train:num_train+num_val]
    test_idx = indices[num_train + num_val:]

    return (images[train_idx], masks[train_idx],
            images[val_idx], masks[val_idx],
            images[test_idx], masks[test_idx]
            )
