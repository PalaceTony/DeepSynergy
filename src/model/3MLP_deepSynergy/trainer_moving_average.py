import torch
import os
import numpy as np

from lib.metrics import mse, rmse, mae, pearson


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        logger,
        cfg,
        train_loader,
        val_loader,
        test_loader,
        epo,
        train_flag,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.logger = logger
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = epo
        self.train_flag = train_flag
        if self.cfg.model.best_path is not None:
            self.best_path = self.cfg.model.best_path
        else:
            self.best_path = os.path.join(os.getcwd(), "best_model.pth")

    def train_epoch(self):
        self.model.train()
        train_loss = 0
        for big_batch in self.train_loader:
            drug_A, drug_B, cell_line, labels = big_batch
            self.optimizer.zero_grad()
            outputs = self.model(drug_A, drug_B, cell_line)
            loss = self.criterion(outputs.squeeze(), labels)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        return train_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for big_batch in self.val_loader:
                drug_A, drug_B, cell_line, labels = big_batch
                outputs = self.model(drug_A, drug_B, cell_line)
                val_loss += self.criterion(outputs.squeeze(), labels)
        return val_loss / len(self.val_loader)

    def train(self):
        val_losses = []
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            val_losses.append(val_loss.cpu())
            self.logger.info(self.train_flag)
            self.logger.info(
                f"Epoch {epoch}/{self.epochs}, Train loss: {train_loss:.4f}, Validate loss: {val_loss:.4f}"
            )
        self.logger.info("-------------------Training completed-------------------")
        if self.train_flag == "second_train":
            torch.save(self.model.state_dict(), self.best_path)
            self.logger.info(f"The best model is found at epoch {self.epochs}")
            self.logger.info(f"Model saved at {self.best_path}")
            self.test()
        return val_losses

    def test(self):
        if self.cfg.model.best_path is not None:
            self.model.load_state_dict(torch.load(self.best_path))
        self.model.eval()
        test_loss = 0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for big_batch in self.test_loader:
                drug_A, drug_B, cell_line, labels = big_batch
                outputs = self.model(drug_A, drug_B, cell_line)
                test_loss += self.criterion(outputs.squeeze(), labels).item()
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
