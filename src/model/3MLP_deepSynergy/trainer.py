import numpy as np
import torch
import torch.nn.functional as F

from lib.utils import moving_average


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        logger,
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
        self.logger = logger

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
                self.logger.info(
                    f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )
            else:
                if epoch == 0:
                    self.logger.info("Testing")
                    self.logger.info(f"The best epoch is {epochs}")
                test_result = self.test()
                test_losses.append(
                    test_result["test_loss"]
                )  # Append the numeric test_loss
                self.logger.info(
                    f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_result['test_loss']:.4f}"
                )
                if epoch == epochs - 1:
                    # Save the model
                    torch.save(self.model.state_dict(), "best_model.pth")
                    self.logger.info("The best model is saved!")

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
            return test_losses

    def test(self):
        self.model.eval()
        test_loss = 0
        total_mse = 0  # To accumulate MSE
        total_samples = 0  # To keep track of the total number of samples

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)

                batch_loss = self.criterion(outputs.squeeze(), targets)
                test_loss += batch_loss.item()

                mse = F.mse_loss(outputs.squeeze(), targets, reduction="sum")
                total_mse += mse.item()

                # Update total samples
                total_samples += targets.size(0)

        # Compute final MSE and RMSE
        final_mse = total_mse / total_samples
        final_rmse = torch.sqrt(torch.tensor(final_mse))

        # Returning the computed metrics
        return {
            "test_loss": test_loss / len(self.test_loader),
            "mse": final_mse,
            "rmse": final_rmse.item(),
        }
