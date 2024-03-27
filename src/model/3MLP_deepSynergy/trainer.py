import torch
import os


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
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.logger = logger
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
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
        best_loss = float("inf")
        not_improved_count = 0

        for epoch in range(1, self.cfg.training.epochs + 1):
            self.logger.info(f"Epoch {epoch}/{self.cfg.training.epochs}")
            _ = self.train_epoch()
            val_loss = self.validate()

            if val_loss < best_loss:
                best_loss = val_loss
                not_improved_count = 0
                torch.save(self.model.state_dict(), self.best_path)
                self.logger.info("Validation loss improved. Saving current best model.")
            else:
                not_improved_count += 1
                self.logger.info(
                    f"Validation loss did not improve. Count: {not_improved_count}/{self.cfg.early_stop_patience}"
                )
            if not_improved_count >= self.cfg.training.early_stop_patience:
                self.logger.info("Early stopping triggered.")
                break

        self.logger.info(f"Training completed. Best model saved to {self.best_path}")
        self.logger.info("Testing NOW")
        self.model.load_state_dict(torch.load(self.best_path))
        self.test()

    def test(self):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for big_batch in self.val_loader:
                drug_A, drug_B, cell_line, labels = big_batch
                outputs = self.model(drug_A, drug_B, cell_line)
                test_loss += self.criterion(outputs.squeeze(), labels).item()
        return test_loss / len(self.val_loader)
