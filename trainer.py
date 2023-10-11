import torch
from torch import nn


class UNETTrainer:
    """
    Trainer class for training a UNet model using PyTorch.

    Args:
        model (torch.nn.Module): The UNet model to be trained.
        train_dataloader (torch.utils.data.DataLoader): Dataloader for training data.
        val_dataloader (torch.utils.data.DataLoader): Dataloader for validation data.

    Attributes:
        model (torch.nn.Module): The UNet model to be trained.
        train_dataloader (torch.utils.data.DataLoader): Dataloader for training data.
        val_dataloader (torch.utils.data.DataLoader): Dataloader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer used for training the model.
        loss_function (torch.nn.Module): Loss function used for training the model.
        num_epochs (int): Number of epochs for training the model.

    Methods:
        train_model():
            Trains the UNet model using the specified training and validation dataloaders.

    Usage Example:
        trainer = UNETTrainer(model, train_dataloader, val_dataloader)
        trained_model, train_losses, train_accs, val_losses, val_accs = trainer.train_model()
    """

    def __init__(self, model: nn.Module, train_dataloader: torch.utils.data.DataLoader, val_dataloader: torch.utils.data.DataLoader):
        """
        Initializes the UNETTrainer object.

        Args:
            model (torch.nn.Module): The UNet model to be trained.
            train_dataloader (torch.utils.data.DataLoader): Dataloader for training data.
            val_dataloader (torch.utils.data.DataLoader): Dataloader for validation data.
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_function = nn.MSELoss()


    def train_model(self, num_epochs: int=1):
        """
        Trains the UNet model using the specified training and validation dataloaders.

        Returns:
            torch.nn.Module: Trained UNet model.
            list: List of training losses for each epoch.
            list: List of training accuracies for each epoch.
            list: List of validation losses for each epoch.
            list: List of validation accuracies for each epoch.
        """
        print("Starting training")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on device: {device}")
        self.model.to(device)
        train_losses, train_accs, val_losses, val_accs = [], [], [], []

        for epoch in range(1, num_epochs + 1):
            model, train_loss, train_acc = self._train_epoch(self.model,
                                                             self.optimizer,
                                                             self.loss_function,
                                                             self.train_dataloader,
                                                             self.val_dataloader,
                                                             device,
                                                             print_every=None,
            )
            val_loss, val_acc = validate(model, loss_fn, val_loader, device)
            print(
                f"Epoch {epoch}/{num_epochs}: "
                f"Train loss: {sum(train_loss)/len(train_loss):.3f}, "
                f"Train acc.: {sum(train_acc)/len(train_acc):.3f}, "
                f"Val. loss: {val_loss:.3f}, "
                f"Val. acc.: {val_acc:.3f}"
            )
            train_losses.extend(train_loss)
            train_accs.extend(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
        return model, train_losses, train_accs, val_losses, val_accs

    def _train_epoch(model, optimizer, loss_fn, train_loader, val_loader, device, print_every):
        # Train:
        model.train()
        train_loss_batches, train_acc_batches = [], []
        num_batches = len(train_loader)
        for batch_index, (x, y) in enumerate(train_loader, 1):
            inputs, labels = x.to(device), y.to(device)
            optimizer.zero_grad()
            z = model.forward(inputs)
            loss = loss_fn(z, labels.float())
            loss.backward()
            optimizer.step()
            train_loss_batches.append(loss.item())

            hard_preds = output_to_label(z)
            acc_batch_avg = (hard_preds == labels).float().mean().item()
            train_acc_batches.append(acc_batch_avg)

            # If you want to print your progress more often than every epoch you can
            # set `print_every` to the number of batches you want between every status update.
            # Note that the print out will trigger a full validation on the full val. set => slows down training
            if print_every is not None and batch_index % print_every == 0:
                val_loss, val_acc = validate(model, loss_fn, val_loader, device)
                model.train()
                print(f"\tBatch {batch_index}/{num_batches}: "
                    f"\tTrain loss: {sum(train_loss_batches[-print_every:])/print_every:.3f}, "
                    f"\tTrain acc.: {sum(train_acc_batches[-print_every:])/print_every:.3f}, "
                    f"\tVal. loss: {val_loss:.3f}, "
                    f"\tVal. acc.: {val_acc:.3f}")

        return model, train_loss_batches, train_acc_batches
