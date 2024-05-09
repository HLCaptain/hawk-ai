import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import ConvNextModel
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback
import webp
from sklearn.model_selection import train_test_split
from lightly.transforms.utils import IMAGENET_NORMALIZE
import xml.etree.ElementTree as ET
import re
import math
import optuna

class NaturnessDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.labels = labels
        self.transform = transform
        # Load images
        # self.images = [webp.load_image(path).convert('RGB') for path in image_paths]
        self.images = images
        # Transform images
        if transform:
            self.images = [transform(image) for image in self.images]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

class NaturenessRegressionModel(LightningModule):
    def __init__(self, backbone, freeze_backbone=False, optimizer=None, scheduler=None):
        super().__init__()
        self.backbone = backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.fc = nn.Linear(768, 1)
        self.criterion = nn.MSELoss()
        self.optimizer = optimizer
        self.scheduler = scheduler

    def forward(self, x):
        x = self.backbone(x).pooler_output
        return self.fc(x).squeeze()

    def step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optim = self.optimizer or torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = self.scheduler or torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=100)
        return [optim], [scheduler]

def parse_xml_for_bndbox_areas(xml_file):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Dictionary to hold the object names and their bounding box areas
    bndbox_areas = {}

    filename = root.find('filename').text

    # Iterate through each object in the XML
    for object_tag in root.findall('object'):
        name = object_tag.find('name').text
        bndbox = object_tag.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # Calculate the area of the bounding box
        area = (xmax - xmin) * (ymax - ymin)

        # Store the area using the name as the key
        cleaned_name = re.sub(r'_\d+$', '', name)
        if name not in bndbox_areas:
            bndbox_areas[cleaned_name] = area
        else:
            bndbox_areas[cleaned_name] += area

    return (filename, bndbox_areas)

def calculate_natureness_score(bndbox_areas: dict[str, int]):
    # Assign natureness value to each object in [-1, 1] range 1 being most natural and -1 being least natural
    # 'SKY': 3747,
    # 'BUILDINGS': 30395,
    # 'POLE': 53873,
    # 'NATURE': 40554,
    # 'GUARD_RAIL': 11840,
    # 'OBSTACLES': 9234,
    # '': 4796,
    # 'CAR': 3811,
    # 'WATER': 2022,
    # 'SIDEWALK': 2619,
    # 'STREET': 3331,
    # 'ROCK': 2405,
    # 'SAND': 934,
    # 'TRAFFIC_SIGNS': 656,
    # 'ROADBLOCK': 4746,
    # 'SOLID_LINE': 3,
    # 'RESTRICTED_STREET': 18
    weights = {
        'SKY': 1.0,
        'BUILDINGS':-1.0,
        'POLE': -0.5,
        'NATURE': 1.0,
        'GUARD_RAIL': -0.5,
        'OBSTACLES': -0.5,
        'CAR': -0.5,
        'WATER': 1.0,
        'SIDEWALK': -0.8,
        'STREET': -0.2,
        'ROCK': 1.0,
        'SAND': 1.0,
        'TRAFFIC_SIGNS': -0.4,
        'ROADBLOCK': 0.2,
        'SOLID_LINE': -0.1,
        'RESTRICTED_STREET': 0.1
    }
    # Split name by last '_' and get the first part
    # Remove names which are not in weights
    bndbox_areas = {name: area for name, area in bndbox_areas.items() if name in weights}
    bndbox_areas_sum = sum(bndbox_areas.values())
    # Calculate the natureness score with atan function in [0, 1] range
    total_weight = sum([weights[name] * area for name, area in bndbox_areas.items()])
    natureness_score = (math.atan(total_weight / bndbox_areas_sum) + math.pi / 2) / math.pi
    return natureness_score

def load_data(data_dir):
    image_paths = glob.glob(os.path.join(data_dir, '**', '*.webp'), recursive=True)
    xml_paths = glob.glob(os.path.join(data_dir, '**', '*.xml'), recursive=True)
    labels = []
    invalid_image_indices = []
    for index, xml_path in enumerate(xml_paths):
        filename, bndbox_areas = parse_xml_for_bndbox_areas(xml_path)
        # Find filename in image_paths and get the index (image_path can be in subdirectories)
        # index = [i for i, path in enumerate(image_paths) if path.endswith(filename)][0]
        try:
            labels.append(calculate_natureness_score(bndbox_areas))
        except ZeroDivisionError:
            invalid_image_indices.append(index)
    # Remove invalid images
    for index in sorted(invalid_image_indices, reverse=True):
        image_paths.pop(index)
    print(f"Loaded {len(image_paths)} images and {len(labels)} labels (removed {len(invalid_image_indices)} invalid images)")
    labels = np.array(labels).astype(np.float32)
    return np.array(image_paths), labels

def create_dataloaders(train_data, val_data, test_data, batch_size, num_workers):
    train_transform = transforms.Compose([
        transforms.RandomCrop(896, padding=4), # Random crop to 896x896
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(
        #     mean=IMAGENET_NORMALIZE["mean"],
        #     std=IMAGENET_NORMALIZE["std"],
        # ),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(
        #     mean=IMAGENET_NORMALIZE["mean"],
        #     std=IMAGENET_NORMALIZE["std"],
        # ),
    ])

    datasets = {
        'train': NaturnessDataset(*train_data, transform=train_transform),
        'val': NaturnessDataset(*val_data, transform=test_transform),
        'test': NaturnessDataset(*test_data, transform=test_transform)
    }

    return {
        'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, persistent_workers=True, pin_memory=True),
        'val': DataLoader(datasets['val'], batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True, persistent_workers=True, pin_memory=True),
        'test': DataLoader(datasets['test'], batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True, persistent_workers=True, pin_memory=True)
    }

class TrialReportCallback(Callback):
    def __init__(self, trial):
        self.trial = trial

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.trial.report(trainer.callback_metrics['val_loss'], trainer.current_epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()

def train_with_trial(trial, dataloader, model_type):
    backbone = ConvNextModel.from_pretrained("facebook/convnext-tiny-224")
    trainer = Trainer(max_epochs=50, callbacks=[EarlyStopping(monitor='val_loss', patience=5), ModelCheckpoint(dirpath='checkpoints/', filename=model_type + '-{val_loss:.2f}', save_top_k=1), TrialReportCallback(trial)], num_sanity_val_steps=0)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    optimizer = torch.optim.Adam(lr=learning_rate, params=backbone.parameters())
    t_max = trial.suggest_int('t_max', 50, 200)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    model = NaturenessRegressionModel(backbone, optimizer=optimizer, scheduler=scheduler)
    trainer.fit(model, dataloader['train'], dataloader['val'])
    test = trainer.test(model, dataloader['test'])
    print(test)
    return test[0]['test_loss']

def main():
    data_dir = '../data'

    num_workers = os.cpu_count() or 1

    all_image_paths, all_labels = load_data(data_dir)
    all_images = np.array([webp.load_image(path).convert('RGB') for path in all_image_paths])

    # Choose a primary threshold based on percentiles or fixed value
    primary_threshold = np.percentile(all_labels, 50)  # 50th percentile or 0.5

    # Probabilities for selection in nature group
    # You can adjust the steepness and center of this sigmoid function
    nature_probabilities = 1 / (1 + np.exp(-25 * (all_labels - primary_threshold)))

    # Randomly select indices based on calculated probabilities
    nature_mask = np.random.rand(*all_labels.shape) < nature_probabilities
    urban_mask = ~nature_mask

    nature_indices = np.where(nature_mask)[0]
    urban_indices = np.where(urban_mask)[0]

    plt.hist(all_labels[nature_indices], bins=100, alpha=0.5, label='Nature')
    plt.hist(all_labels[urban_indices], bins=100, alpha=0.5, label='Urban')
    plt.legend()
    plt.show()

    nature_images = all_images[nature_indices]
    nature_labels = all_labels[nature_indices]
    urban_images = all_images[urban_indices]
    urban_labels = all_labels[urban_indices]

    dataloaders = {}
    model_types = ['all', 'nature', 'urban']
    images_and_labels = [(all_images, all_labels), (nature_images, nature_labels), (urban_images, urban_labels)]

    for model_type, (images, labels) in zip(model_types, images_and_labels):
        dataloaders[model_type] = {}
        for batch_size in [16, 32, 64]:
            x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
            x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
            dataloaders[model_type][batch_size] = create_dataloaders((x_train, y_train), (x_val, y_val), (x_test, y_test), batch_size, num_workers)

    models = {}

    # trainer.fit(models['all'], dataloaders['all']['train'], dataloaders['all']['val'])
    # trainer.fit(models['nature'], dataloaders['nature']['train'], dataloaders['nature']['val'])
    # trainer.fit(models['urban'], dataloaders['urban']['train'], dataloaders['urban']['val'])

    def train_all(trial):
        dataloader = dataloaders['all'][trial.suggest_categorical('batch_size', [16, 32, 64])]
        return train_with_trial(trial, dataloader, 'all')

    def train_nature(trial):
        dataloader = dataloaders['nature'][trial.suggest_categorical('batch_size', [16, 32, 64])]
        return train_with_trial(trial, dataloader, 'nature')

    def train_urban(trial):
        dataloader = dataloaders['urban'][trial.suggest_categorical('batch_size', [16, 32, 64])]
        return train_with_trial(trial, dataloader, 'urban')

    torch.set_float32_matmul_precision('medium')
    for objective in [train_all, train_nature, train_urban]:
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=25)
        print(f'Study value: {study.best_value}\nStudy best params: {study.best_params}')

    # Eval
    # trainer.test(models['all'], dataloaders['all']['test'])
    # trainer.test(models['nature'], dataloaders['nature']['test'])
    # trainer.test(models['urban'], dataloaders['urban']['test'])

    for model_type in model_types:
        # Get all model checkpoint paths and load the best one
        best_ckpt_path= None
        best_loss = None
        for checkpoint_path in glob.glob(f'checkpoints/{model_type}-*.ckpt'):
            # * should be the lowest loss value
            new_loss = float(checkpoint_path.split('=')[-1].replace('.ckpt', ''))
            if best_loss is None or new_loss < best_loss:
                best_ckpt_path = checkpoint_path
                best_loss = new_loss
        models[model_type] = LightningModule.load_from_checkpoint(best_ckpt_path)

    # Compare predicted and actual values
    for model_type, model in models.items():
        # Save the model
        torch.save(model.state_dict(), f'natureness_model_{model_type}.pth')
        model.eval()
        for m_type in model_types:
            # Show example images with their predicted and actual values
            for i in range(4):
                x, y = dataloaders[m_type]['test'].dataset[i]
                y_pred = model(x.unsqueeze(0)).item()
                plt.imshow(x.permute(1, 2, 0))
                plt.title(f"Model type: {model_type}, Dataset: {m_type}\nPredicted: {y_pred:.2f}, Actual: {y:.2f}")
                plt.show()
                plt.savefig(f"example_{i}_model_{model_type}_data_{m_type}.png")

if __name__ == '__main__':
    main()
