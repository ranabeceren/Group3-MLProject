import numpy as np

def train_val_test_split(images,
                         masks,
                         train=0.7,
                         test=0.2,
                         val=0.1,
                         seed=42):
    
    np.random.seed(seed)

    indicies = np.arange(len(images))
    np.random.shuffle(indicies)

    num_train = int(len(indicies) * train)
    num_val = int(len(indicies) * val)

    train_idx = indicies[:num_train]
    test_idx = indicies[num_train:num_train+num_val]
    val_idx = indicies[:num_train]

    return (images[train_idx], masks[train_idx],
            images[val_idx], masks[val_idx],
            images[test_idx], masks[test_idx],
            )
