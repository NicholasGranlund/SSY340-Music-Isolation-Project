import torch
from torch import nn
import datetime as dt
import os


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

    def __init__(self, model: nn.Module, train_dataloader: torch.utils.data.DataLoader,
                 val_dataloader: torch.utils.data.DataLoader):
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

    def train_model(self, num_epochs: int = 1):
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

        # To save the model at each epoch
        os.makedirs('training_model', exist_ok=True)
        training_id = dt.datetime.now().strftime('%Y_%m_%d_%H_%M')
        training_folder = f'training_model/{training_id}'
        os.makedirs(training_folder, exist_ok=True)

        for epoch in range(1, num_epochs + 1):
            model, train_loss_batches, train_acc_batches = self._train_epoch(device, print_every=None)
            val_loss, val_acc = self.validate(device)
            print(
                f"Epoch {epoch}/{num_epochs}: "
                f"Train loss: {sum(train_loss_batches) / len(train_loss_batches):.3f}, "
                f"Train acc.: {sum(train_acc_batches) / len(train_acc_batches):.3f}, "
                f"Val. loss: {val_loss:.3f}, "
                f"Val. acc.: {val_acc:.3f}"
            )
            train_losses.append(sum(train_loss_batches) / len(train_loss_batches))
            train_accs.append(sum(train_acc_batches) / len(train_acc_batches))
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            torch.save({'model_state_dict': self.model.state_dict(),
                        'train_losses': train_losses,
                        'train_accs': train_accs,
                        'val_losses': val_losses,
                        'val_accs': val_accs,
                        }, os.path.join(training_folder, f'epoch_{epoch}.ckpt'))

        return self.model, train_losses, train_accs, val_losses, val_accs

    def _train_epoch(self, device, print_every):
        # Train:
        self.model.train()
        train_loss_batches, train_acc_batches = [], []
        num_batches = len(self.train_dataloader)
        for batch_index, (x, y) in enumerate(self.train_dataloader, 1):
            inputs, labels = x.to(device), y.to(device)
            self.optimizer.zero_grad()
            z = self.model.forward(inputs)
            loss = self.loss_function(z, labels.float())
            loss.backward()
            self.optimizer.step()
            train_loss_batches.append(loss.item())

            hard_preds = self.output_to_label(z)
            acc_batch_avg = (hard_preds == labels).float().mean().item()
            train_acc_batches.append(acc_batch_avg)

            # If you want to print your progress more often than every epoch you can
            # set `print_every` to the number of batches you want between every status update.
            # Note that the print out will trigger a full validation on the full val. set => slows down training
            if print_every is not None and batch_index % print_every == 0:
                val_loss, val_acc = self.validate(device)
                self.model.train()
                print(f"\tBatch {batch_index}/{num_batches}: "
                      f"\tTrain loss: {sum(train_loss_batches[-print_every:]) / print_every:.3f}, "
                      f"\tTrain acc.: {sum(train_acc_batches[-print_every:]) / print_every:.3f}, "
                      f"\tVal. loss: {val_loss:.3f}, "
                      f"\tVal. acc.: {val_acc:.3f}")

        return self.model, train_loss_batches, train_acc_batches

    def output_to_label(z: torch.tensor):
        """
        Args:
            z (torch.tensor): ouput of the network
        Returns:
            The binary mask of the prediction
        """
        return torch.where(z > 0.5, 1, 0, dtype=torch.int)

    def validate(self, device):

        val_loss_cum = 0
        val_acc_cum = 0
        self.model.eval()
        with torch.no_grad():
            for batch_index, (x, y) in enumerate(self.val_dataloader, 1):
                inputs, labels = x.to(device), y.to(device)
                z = self.model.forward(inputs)

                batch_loss = self.loss_function(z, labels.float())
                val_loss_cum += batch_loss.item()
                hard_preds = self.output_to_label(z)
                acc_batch_avg = (hard_preds == labels).float().mean().item()
                val_acc_cum += acc_batch_avg
        return val_loss_cum / len(self.val_dataloader), val_acc_cum / len(self.val_dataloader)
