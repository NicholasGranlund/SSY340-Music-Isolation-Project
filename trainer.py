import torch
from torch import nn
import datetime as dt
import os
from metric import dice_coeff


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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on device: {device}")
        self.model.to(device)
        train_losses, train_accs, val_losses, val_accs = [], [], [], []

        # To save the model at each epoch
        os.makedirs('training_model', exist_ok=True)
        training_id = dt.datetime.now().strftime('%Y_%m_%d_%H_%M')
        training_folder = f'training_model/{training_id}'
        os.makedirs(training_folder, exist_ok=True)

        # Iterate through the epochs
        for epoch in range(1, num_epochs + 1):

            # Train one epoch
            improved_model, train_loss_batches, train_accs_batches = self._train_epoch(device, print_every=None)

            # Save model
            self.model = improved_model
            train_loss_batches.append(sum(train_loss_batches)/len(train_loss_batches))
            train_accs_batches.append(sum(train_accs_batches)/len(train_accs_batches))
            # Save losses
            self.epoch_losses.append(train_loss_batches[-1])
            print(f'Epoch number {epoch} complete: Average loss: {self.epoch_losses[-1]} Accuracy: {train_accs_batches[-1]}')

        return self.model, train_losses, train_accs, val_losses, val_accs

    def _train_epoch(self, device, print_every):
        
        # Set the model in train mode
        self.model.train()

        # Initialize empy lists for losses
        train_loss_batches, train_acc_batches = [], []

        # Get number of batches 
        num_batches = len(self.train_dataloader)

        # Iterate thought the batches
        for batch_index, (x, y) in enumerate(self.train_dataloader, 1):
            print(f"\tBatch {batch_index}/{num_batches}: ")

            # Send tensors to device
            inputs_spectrogram, outputs_mask = x.to(device), y.to(device)
            self.optimizer.zero_grad()

            # Forward prop, get mask prediction
            mask_prediction = self.model.forward(inputs_spectrogram)
 
            # Compute the loss
            loss = self.loss_function(mask_prediction, outputs_mask)

            # Back propagate
            loss.backward()
            self.optimizer.step()


            hard_preds = self.output_to_label(mask_prediction)
            acc_batch_avg = (hard_preds == outputs_mask).float().mean().item()
            train_acc_batches.append(acc_batch_avg)

            # Append the loss in the list
            train_loss_batches.append(loss.item())
            print(f'Loss: {loss.item()}')

        return self.model, train_loss_batches, train_acc_batches

    def validate(self, device):

        val_loss_cum = 0
        val_acc_cum = []
        self.model.eval()
        with torch.no_grad():
            for batch_index, (x, y) in enumerate(self.val_dataloader, 1):
                inputs, labels = x.to(device), y.to(device)
                z = self.model.forward(inputs)

                batch_loss = self.loss_function(z, labels.float())
                val_loss_cum += batch_loss.item()
                hard_preds = self.output_to_label(z)
                acc_batch_avg = dice_coeff(hard_preds, labels)
                val_acc_cum.append(acc_batch_avg)
        return val_loss_cum / len(self.val_dataloader), sum(val_acc_cum) / len(val_acc_cum)

    @staticmethod
    def output_to_label(pred: torch.tensor):
        return torch.where(pred > 0.5, 1, 0)
