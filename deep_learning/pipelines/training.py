import torch
import torch.nn as nn
from deep_learning.trainers.trainerV2 import train_step, test_step
from deep_learning.metrics.train_metrics import compute_pos_weight, print_train_time
from torchmetrics.classification import BinaryAccuracy
from timeit import default_timer as timer
from tqdm.auto import tqdm


def training(
    train_loader,
    test_loader,
    val_loader,

    model,
    epochs, 
    learning_rate):

    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Choose loss_fn, optimizer and metrics
    pos_weight = compute_pos_weight(train_loader, device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    accuracy_fn = BinaryAccuracy(threshold=0.5).to(device)

    # Reduce LR on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",        # because we monitor val loss
    factor=0.99,        # lr = lr * 0.99
    patience=15,        # wait 15 epochs with no improvement
    #verbose=True
    )
    # Randomize for reproducability
    RANDOM_SEED = 42
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED) # because our code runs on GPU

    # Measure time
    start = timer()

    # Set epochs
    epochs = epochs

    for epoch in tqdm(range(epochs)): # wraps a progression bar around epochs
        print(f"Epoch: {epoch}\n---------")

        # Train data
        train_step(model=model,
                data_loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                accuracy_fn=accuracy_fn,
                device=device)
        
        # Test data
        test_step(model=model,
                data_loader=test_loader,
                loss_fn=loss_fn,
                accuracy_fn=accuracy_fn,
                scheduler=scheduler,
                device=device)
        
    end = timer()
    train_time = print_train_time(start=start,
                                end=end,
                                device=device)