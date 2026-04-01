import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as L
from torch.utils_data import DataLoader
from models.detector import DeepfakeDetector
from train.dataset import TrainDataset
from torchmetrics import Accuracy, F1Score, AUROC

class DeepfakeModule(L.LightningModule):
    """
    LightningModule for Deepfake Detection.
    """
    def __init__(self, model, lr=1e-4):
        super(DeepfakeModule, self).__init__()
        self.model = model
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='binary')
        self.f1 = F1Score(task='binary')
        self.auc = AUROC(task='binary')
        
    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        # Simulate video input frames
        x = torch.randn(x.size(0), 3, 16, 224, 224) 
        output = self(x)
        loss = self.criterion(output, y)
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        # Simulate video input frames
        x = torch.randn(x.size(0), 3, 16, 224, 224) 
        output = self(x)
        val_loss = self.criterion(output, y)
        val_acc = self.accuracy(output.argmax(dim=1), y)
        val_f1 = self.f1(output.argmax(dim=1), y)
        val_auc = self.auc(output[:, 1], y)
        
        self.log_dict({'val_loss': val_loss, 'val_acc': val_acc, 'val_f1': val_f1, 'val_auc': val_auc})
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss',
        }

if __name__ == "__main__":
    # Test script for training loop
    model = DeepfakeDetector()
    module = DeepfakeModule(model)
    trainer = L.Trainer(max_epochs=10, limit_train_batches=0.1)
    # trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
