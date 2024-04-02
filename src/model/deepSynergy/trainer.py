import numpy as np
import torch
import torch.nn.functional as F
import os

from lib.metrics import mse, rmse, mae, pearson


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        logger,
        args,
        train_loader,
        val_loader,
        test_loader,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.logger = logger
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        if self.args.best_path is not None:
            self.best_path = self.args.best_path
        else:
            self.best_path = os.path.join(self.args.output_dir, "best_model.pth")

    def train_epoch(self):
        self.model.train()
        train_loss = 0
        for inputs, labels in self.train_loader:
            inputs, labels = inputs, labels
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs.squeeze(), labels)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        return train_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs, labels
                outputs = self.model(inputs)
                val_loss += self.criterion(outputs.squeeze(), labels)
        return val_loss / len(self.val_loader)

    def train(self):
        best_loss = float("inf")
        not_improved_count = 0

        for epoch in range(1, self.args.epochs + 1):
            self.logger.info(f"Epoch {epoch}/{self.args.epochs}")
            train_loss = self.train_epoch()
            val_loss = self.validate()

            if val_loss < best_loss:
                best_loss = val_loss
                not_improved_count = 0
                torch.save(self.model.state_dict(), self.best_path)
                self.logger.info(
                    f"Train loss: {train_loss:.4f}, Validate loss: {val_loss:.4f}"
                )
                self.logger.info("Validation loss improved. Saving current best model.")
            else:
                not_improved_count += 1
                self.logger.info(
                    f"Train loss: {train_loss:.4f}, Validate loss: {val_loss:.4f}"
                )
                self.logger.info(
                    f"Validation loss did not improve. Count: {not_improved_count}/{self.args.early_stop_patience}"
                )
            if not_improved_count >= self.args.early_stop_patience:
                self.logger.info("Early stopping triggered.")
                break

        self.logger.info("-------------------Training completed-------------------")
        self.logger.info(f"Best model saved to {self.best_path}")
        self.logger.info("-------------------Testing -------------------")
        self.model.load_state_dict(torch.load(self.best_path))
        self.test()

    def test(self):
        if self.args.best_path is not None:
            self.model.load_state_dict(torch.load(self.best_path))

        self.model.eval()
        test_loss = 0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs, labels
                outputs = self.model(inputs)

                test_loss += self.criterion(outputs.squeeze(), labels)

                y_true.append(labels.cpu().numpy())
                y_pred.append(outputs.squeeze().cpu().numpy())

        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        mse_value = mse(y_true, y_pred)
        rmse_value = rmse(y_true, y_pred)
        mae_value = mae(y_true, y_pred)
        pearson_value = pearson(y_true, y_pred)

        self.logger.info(f"Test MSE: {mse_value:.4f}")
        self.logger.info(f"Test RMSE: {rmse_value:.4f}")
        self.logger.info(f"Test MAE: {mae_value:.4f}")
        self.logger.info(f"Test Pearson: {pearson_value:.4f}")
        self.logger.info(f"Test Loss: {test_loss / len(self.val_loader):.4f}")

        return test_loss / len(self.val_loader)
