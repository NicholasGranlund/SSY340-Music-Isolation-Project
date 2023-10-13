import torch
from torch import nn
import datetime as dt
import os

class UnetTrainer:
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
        self.epoch_losses = []


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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on device: {self.device}")
        self.model.to(self.device)

        train_losses, train_accs, val_losses, val_accs = [], [], [], []

        # To save the model at each epoch
        os.makedirs('training_model', exist_ok=True)
        training_id = dt.datetime.now().strftime('%Y_%m_%d_%H_%M')
        training_folder = f'training_model/{training_id}'
        os.makedirs(training_folder, exist_ok=True)

        # Iterate through the epochs
        for epoch in range(1, num_epochs + 1):

            # Train one epoch
            improved_model, train_loss_batches = self._train_epoch()

            # Save model
            self.model = improved_model

            # Save losses
            self.epoch_losses.append(sum(train_loss_batches)/len(train_loss_batches))

            # Print
            print(f'Epoch {epoch} done.     Mean MSELoss = {self.epoch_losses[-1]}')

        return self.model, self.epoch_losses


    def _train_epoch(self):
        
        # Set the model in train mode
        self.model.train()

        # Initialize empy lists for losses
        train_loss_batches = []

        # Get number of batches 
        num_batches = len(self.train_dataloader)

        # Iterate thought the batches
        for batch_index, (x, y) in enumerate(self.train_dataloader, 1):

            # Send tensors to device
            inputs_spectrogram, outputs_mask = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            # Forward prop, get mask prediction
            mask_prediction = self.model.forward(inputs_spectrogram)
 
            # Compute the loss
            loss = self.loss_function(mask_prediction*inputs_spectrogram, outputs_mask*inputs_spectrogram)

            # Back propagate
            loss.backward()
            self.optimizer.step()

            # Append the loss in the list
            train_loss_batches.append(loss.item())
        
        return self.model, train_loss_batches

