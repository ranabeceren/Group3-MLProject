def compute_pos_weight(dataloader, device):
    pos = 0
    neg = 0

    for _, y in dataloader:
        y = y.to(device)
        y = y.float()
        if y.ndim == 3:
            y = y.unsqueeze(1)

        pos += y.sum()
        neg += (1 - y).sum()

    pos_weight = neg / pos
    return pos_weight