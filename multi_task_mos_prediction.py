#/Users/aryansingh/Downloads/tid2013/distorted_images
# train_multitask_model.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import numpy as np
import os
from torch.utils.data import DataLoader
from load_data import build_dataset
from tqdm import tqdm  # For progress bars
import scipy.stats

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the MultiTaskModel
class MultiTaskModel(nn.Module):
    def __init__(self, num_distortion_classes=24, num_opinion_scores=10):
        super(MultiTaskModel, self).__init__()
        # Shared feature extractor
        self.shared_layers = models.resnet18(pretrained=True)
        num_features = self.shared_layers.fc.in_features
        self.shared_layers.fc = nn.Identity()  # Remove the last layer

        # Distortion type prediction head
        self.distortion_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_distortion_classes)
        )

        # Opinion score distribution prediction head
        self.opinion_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_opinion_scores),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        features = self.shared_layers(x)
        distortion_logits = self.distortion_head(features)
        opinion_distribution = self.opinion_head(features)
        return distortion_logits, opinion_distribution

def emd_loss(target, output):
    p_cumulative = torch.cumsum(output, dim=1)
    q_cumulative = torch.cumsum(target, dim=1)
    emd = torch.sqrt(torch.mean((p_cumulative - q_cumulative) ** 2, dim=1))
    loss = torch.mean(emd)
    return loss

def total_loss_fn(distortion_logits, distortion_labels, opinion_preds, opinion_labels, alpha=1.0, beta=1.0):
    distortion_loss_fn = nn.CrossEntropyLoss()
    distortion_loss = distortion_loss_fn(distortion_logits, distortion_labels)
    opinion_loss = emd_loss(opinion_labels, opinion_preds)
    total_loss = alpha * distortion_loss + beta * opinion_loss
    return total_loss, distortion_loss, opinion_loss

if __name__ == '__main__':
    # Paths (update these paths based on your setup)
    pth_path = 'saved_models/'
    path_to_json_train = 'tid_labels_train.json'
    path_to_json_test = 'tid_labels_test.json'
    path_to_imgs = '/Users/aryansingh/Downloads/tid2013/distorted_images'

    # Build datasets and dataloaders
    TID_train, TID_val, TID_test = build_dataset(path_to_imgs, path_to_json_train, path_to_json_test)

    data_loader_train = DataLoader(TID_train, batch_size=16, num_workers=1, shuffle=True)
    data_loader_val = DataLoader(TID_val, batch_size=16, num_workers=1)
    data_loader_test = DataLoader(TID_test, batch_size=16, num_workers=1)

    # Initialize the model
    model = MultiTaskModel(num_distortion_classes=24, num_opinion_scores=10).to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training parameters
    num_epochs = 50
    alpha = 1.0  # Weight for distortion loss
    beta = 1.0   # Weight for opinion loss

    best_val_loss = float('inf')
    best_epoch = 0

    # Training and validation loops
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0.0
        total_distortion_loss = 0.0
        total_opinion_loss = 0.0

        for batch in tqdm(data_loader_train, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            images = batch['img'].to(device)
            opinion_labels = batch['label'].to(device)
            distortion_labels = batch['distortion_label'].to(device)

            optimizer.zero_grad()

            distortion_logits, opinion_preds = model(images)

            loss, distortion_loss, opinion_loss = total_loss_fn(
                distortion_logits, distortion_labels, opinion_preds, opinion_labels, alpha=alpha, beta=beta
            )

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * images.size(0)
            total_distortion_loss += distortion_loss.item() * images.size(0)
            total_opinion_loss += opinion_loss.item() * images.size(0)

        avg_train_loss = total_train_loss / len(TID_train)
        avg_train_distortion_loss = total_distortion_loss / len(TID_train)
        avg_train_opinion_loss = total_opinion_loss / len(TID_train)

        print(f'Epoch [{epoch+1}/{num_epochs}] Training Loss: {avg_train_loss:.4f}, '
              f'Distortion Loss: {avg_train_distortion_loss:.4f}, '
              f'Opinion Loss: {avg_train_opinion_loss:.4f}')

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        total_val_distortion_loss = 0.0
        total_val_opinion_loss = 0.0
        total_accuracy = 0.0
        mos_preds_list = []
        mos_labels_list = []

        with torch.no_grad():
            for batch in tqdm(data_loader_val, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images = batch['img'].to(device)
                opinion_labels = batch['label'].to(device)
                distortion_labels = batch['distortion_label'].to(device)

                distortion_logits, opinion_preds = model(images)

                loss, distortion_loss, opinion_loss = total_loss_fn(
                    distortion_logits, distortion_labels, opinion_preds, opinion_labels, alpha=alpha, beta=beta
                )

                total_val_loss += loss.item() * images.size(0)
                total_val_distortion_loss += distortion_loss.item() * images.size(0)
                total_val_opinion_loss += opinion_loss.item() * images.size(0)

                # Compute accuracy for distortion type prediction
                _, preds = torch.max(distortion_logits, 1)
                correct = (preds == distortion_labels).sum().item()
                accuracy = correct / images.size(0)
                total_accuracy += accuracy * images.size(0)

                # Compute MOS predictions and labels
                mos_preds = torch.sum(opinion_preds * torch.arange(1, 11).float().to(device), dim=1)
                mos_labels = torch.sum(opinion_labels * torch.arange(1, 11).float().to(device), dim=1)

                mos_preds_list.extend(mos_preds.cpu().numpy())
                mos_labels_list.extend(mos_labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(TID_val)
        avg_val_distortion_loss = total_val_distortion_loss / len(TID_val)
        avg_val_opinion_loss = total_val_opinion_loss / len(TID_val)
        avg_accuracy = total_accuracy / len(TID_val)

        # Compute SRCC and PCC
        srcc, _ = scipy.stats.spearmanr(mos_preds_list, mos_labels_list)
        pcc, _ = scipy.stats.pearsonr(mos_preds_list, mos_labels_list)

        print(f'Epoch [{epoch+1}/{num_epochs}] Validation Loss: {avg_val_loss:.4f}, '
              f'Distortion Loss: {avg_val_distortion_loss:.4f}, '
              f'Opinion Loss: {avg_val_opinion_loss:.4f}, '
              f'Accuracy: {avg_accuracy:.4f}, SRCC: {srcc:.4f}, PCC: {pcc:.4f}')

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            if not os.path.exists(pth_path):
                os.makedirs(pth_path)
            torch.save(model.state_dict(), os.path.join(pth_path, 'best_multitask_model.pth'))
            print(f'Best model saved at epoch {best_epoch}')

    print('Training completed.')

    # Load the best model for testing
    model.load_state_dict(torch.load(os.path.join(pth_path, 'best_multitask_model.pth')))

    # Testing phase
    model.eval()
    total_test_loss = 0.0
    total_test_distortion_loss = 0.0
    total_test_opinion_loss = 0.0
    total_accuracy = 0.0
    mos_preds_list = []
    mos_labels_list = []

    with torch.no_grad():
        for batch in tqdm(data_loader_test, desc="Testing"):
            images = batch['img'].to(device)
            opinion_labels = batch['label'].to(device)
            distortion_labels = batch['distortion_label'].to(device)

            distortion_logits, opinion_preds = model(images)

            loss, distortion_loss, opinion_loss = total_loss_fn(
                distortion_logits, distortion_labels, opinion_preds, opinion_labels, alpha=alpha, beta=beta
            )

            total_test_loss += loss.item() * images.size(0)
            total_test_distortion_loss += distortion_loss.item() * images.size(0)
            total_test_opinion_loss += opinion_loss.item() * images.size(0)

            # Compute accuracy for distortion type prediction
            _, preds = torch.max(distortion_logits, 1)
            correct = (preds == distortion_labels).sum().item()
            accuracy = correct / images.size(0)
            total_accuracy += accuracy * images.size(0)

            # Compute MOS predictions and labels
            mos_preds = torch.sum(opinion_preds * torch.arange(1, 11).float().to(device), dim=1)
            mos_labels = torch.sum(opinion_labels * torch.arange(1, 11).float().to(device), dim=1)

            mos_preds_list.extend(mos_preds.cpu().numpy())
            mos_labels_list.extend(mos_labels.cpu().numpy())

    avg_test_loss = total_test_loss / len(TID_test)
    avg_test_distortion_loss = total_test_distortion_loss / len(TID_test)
    avg_test_opinion_loss = total_test_opinion_loss / len(TID_test)
    avg_accuracy = total_accuracy / len(TID_test)

    # Compute SRCC and PCC
    srcc, _ = scipy.stats.spearmanr(mos_preds_list, mos_labels_list)
    pcc, _ = scipy.stats.pearsonr(mos_preds_list, mos_labels_list)

    print(f'Test Loss: {avg_test_loss:.4f}, '
          f'Distortion Loss: {avg_test_distortion_loss:.4f}, '
          f'Opinion Loss: {avg_test_opinion_loss:.4f}, '
          f'Accuracy: {avg_accuracy:.4f}, SRCC: {srcc:.4f}, PCC: {pcc:.4f}')
