import numpy as np

def train_test_val_split(images,
                         masks,
                         train=0.7,
                         test=0.2,
                         seed=42):
    
    np.random.seed(seed)

    indices = np.arange(len(images))
    np.random.shuffle(indices)

    num_train = int(len(indices) * train)
    num_test = int(len(indices) * test)

    train_idx = indices[:num_train]
    test_idx = indices[num_train:num_train+num_test]
    val_idx = indices[num_train + num_test:]

    return (images[train_idx], masks[train_idx],
            images[test_idx], masks[test_idx],
            images[val_idx], masks[val_idx],
            )
