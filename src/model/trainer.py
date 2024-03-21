import numpy as np
import torch

from lib.utils import moving_average


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        train_loader,
        device,
        cfg,
        val_loader=None,
        test_loader=None,
        firstTrain=True,
        epochs=None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.cfg = cfg
        self.firstTrain = firstTrain
        self.epochs = epochs

    def train_epoch(self):
        self.model.train()
        train_loss = 0
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs.squeeze(), targets)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        return train_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                val_loss += self.criterion(outputs.squeeze(), targets).item()
        return val_loss / len(self.val_loader)

    def train(self):
        val_losses = []
        test_losses = []
        epochs = self.epochs if self.epochs is not None else self.cfg.training.epochs
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            if self.firstTrain:
                val_loss = self.validate()
                val_losses.append(val_loss)
                print(
                    f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )
            else:
                print("Testing") if epoch == 0 else None
                test_loss = self.test()
                test_losses.append(test_loss)
                print(
                    f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}"
                )

        if self.firstTrain:
            # Determine early stopping point
            average_over = self.cfg.training.early_stopping_average_over
            mov_av = moving_average(np.array(val_losses), average_over)
            smooth_val_loss = np.pad(
                mov_av, (average_over // 2, average_over // 2), mode="edge"
            )
            epo = np.argmin(smooth_val_loss)
            return epo, val_losses, smooth_val_loss
        else:
            return None, None, None

    def test(self):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                test_loss += self.criterion(outputs.squeeze(), targets).item()
        return test_loss / len(self.test_loader)
