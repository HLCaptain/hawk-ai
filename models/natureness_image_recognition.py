import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import ConvNextModel
from lightning import LightningModule, Trainer, LightningDataModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback
import webp
from sklearn.model_selection import train_test_split
from lightly.transforms.utils import IMAGENET_NORMALIZE
import xml.etree.ElementTree as ET
import re
import math
import optuna
import pandas as pd

class NaturnessDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.labels = labels
        self.transform = transform
        # Load images
        # self.images = [webp.load_image(path).convert('RGB') for path in image_paths]
        self.image_paths = image_paths

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = webp.load_image(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

class NaturenessDataModule(LightningDataModule):
    def __init__(self, image_paths, labels, batch_size, num_workers):
        super().__init__()
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        x_train, x_test, y_train, y_test = train_test_split(self.image_paths, self.labels, test_size=0.2, random_state=42)
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
        self.dataloaders = create_dataloaders((x_train, y_train), (x_val, y_val), (x_test, y_test), self.batch_size, self.num_workers)

    def train_dataloader(self):
        return self.dataloaders['train']
    def val_dataloader(self):
        return self.dataloaders['val']
    def test_dataloader(self):
        return self.dataloaders['test']
class NaturenessRegressionModel(LightningModule):
    def __init__(self, backbone=None, freeze_backbone=False, optimizer=None, scheduler=None):
        super().__init__()
        self.save_hyperparameters(backbone, freeze_backbone, optimizer, scheduler)
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

def train_with_trial(trial, datamodule, model_type):
    backbone = ConvNextModel.from_pretrained("facebook/convnext-tiny-224")
    trainer = Trainer(max_epochs=10, callbacks=[EarlyStopping(monitor='val_loss', patience=5), ModelCheckpoint(dirpath='checkpoints/', filename=model_type + '-{val_loss:.4f}', save_top_k=1), TrialReportCallback(trial)], num_sanity_val_steps=0)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    optimizer = torch.optim.Adam(lr=learning_rate, params=backbone.parameters())
    t_max = trial.suggest_int('t_max', 50, 200)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    model = NaturenessRegressionModel(backbone, optimizer=optimizer, scheduler=scheduler)
    trainer.fit(model, datamodule=datamodule)
    test = trainer.test(model, datamodule=datamodule)
    torch.save(model.state_dict(), f'natureness_model_{model_type}.pth') # Save the newest model
    print(test)
    return test[0]['test_loss'], model

def train_eval_save(image_paths, labels, batch_sizes=[16, 32, 64]):
    num_workers = os.cpu_count() or 1
    datamodules = {}
    for batch_size in batch_sizes:
        datamodules[batch_size] = NaturenessDataModule(image_paths, labels, batch_size, num_workers)
    study = optuna.create_study(direction='minimize')
    def train_all(trial):
        global all_model, all_model_loss
        datamodule = datamodules[trial.suggest_categorical('batch_size', batch_sizes)]
        loss, model = train_with_trial(trial, datamodule, 'all')
        if all_model_loss is None or loss < all_model_loss:
            all_model = model
            all_model_loss = loss
        return loss
    torch.set_float32_matmul_precision('medium')
    study.optimize(train_all, n_trials=1)
    print(f'Study value: {study.best_value}\nStudy best params: {study.best_params}')
    return all_model, datamodules

def train_eval_save_nature_urban(all_image_paths, all_labels, percentile=50, uniformity=25):
    # Choose a primary threshold based on percentiles or fixed value
    primary_threshold = np.percentile(all_labels, percentile)  # 50th percentile or 0.5

    # Probabilities for selection in nature group
    # You can adjust the steepness and center of this sigmoid function
    nature_probabilities = 1 / (1 + np.exp(-uniformity * (all_labels - primary_threshold)))

    # Randomly select indices based on calculated probabilities
    nature_mask = np.random.rand(*all_labels.shape) < nature_probabilities
    urban_mask = ~nature_mask

    nature_indices = np.where(nature_mask)[0]
    urban_indices = np.where(urban_mask)[0]

    # plt.hist(all_labels[nature_indices], bins=100, alpha=0.5, label='Nature')
    # plt.hist(all_labels[urban_indices], bins=100, alpha=0.5, label='Urban')
    # plt.legend()
    # plt.show()

    nature_images = [all_image_paths[i] for i in nature_indices]
    nature_labels = all_labels[nature_indices]
    urban_images = [all_image_paths[i] for i in urban_indices]
    urban_labels = all_labels[urban_indices]

    datamodules = {}
    model_types = ['nature', 'urban']
    images_and_labels = [(nature_images, nature_labels), (urban_images, urban_labels)]

    torch.set_float32_matmul_precision('medium')

    models = {}
    for model_type, (images, labels) in zip(model_types, images_and_labels):
        models[model_type], datamodules[model_type] = train_eval_save(images, labels)

    return models, datamodules

all_model = None
all_model_loss = None
urban_model = None
urban_model_loss = None
nature_model = None
nature_model_loss = None
models = {}
models_loss = {}

def main():
    data_dir = '../data'

    all_image_paths, all_labels = load_data(data_dir)
    # all_images = [webp.load_image(path).convert('RGB') for path in all_image_paths]

    all_models = {}
    all_datamodules = {}
    uniformities = [10, 25, 40]
    percentiles = [30, 50, 70]

    # Train single model for all data
    all_model, all_datamodule = train_eval_save(all_image_paths, all_labels)
    all_models['all'] = {'all': all_model}
    all_datamodules['all'] = {'all': all_datamodule}
    print(f"all_model: {all_model}, all_datamodule: {all_datamodule}")

    # Train models for nature and urban groups
    for uniformity in uniformities:
        for percentile in percentiles:
            print(f"Training with percentile: {percentile} and uniformity: {uniformity}")
            models, datamodules = train_eval_save_nature_urban(all_image_paths, all_labels, percentile, uniformity)
            config_key = f'p{percentile}_u{uniformity}'
            all_models[config_key] = models
            datamodules['all'] = all_datamodule
            all_datamodules[config_key] = datamodules
            print(f"models: {models}, datamodules: {datamodules}")

    # Example usage of all_models and all_datamodules
    dataset_types = ['all', 'nature', 'urban']
    model_losses = {}
    for config_key, models in all_models.items():
        for model_type, model in models.items():
            datamodules = all_datamodules[config_key]
            print(f'Datamodules: {datamodules}')
            for dataset_type in dataset_types:
                model_losses[(config_key, model_type, dataset_type)] = []
                for i in range(16):
                    datamodules[dataset_type][16].setup()
                    x, y = datamodules[dataset_type][16].test_dataloader().dataset[i]
                    y_pred = model(x.unsqueeze(0)).item()
                    model_losses[(config_key, model_type, dataset_type)].append(y_pred)
                    plt.imshow(x.permute(1, 2, 0))
                    plt.title(f"Config: {config_key}, Model type: {model_type}, Dataset: {dataset_type}\nPredicted: {y_pred:.2f}, Actual: {y:.2f}")
                    plt.savefig(f"example_{i}_{config_key}_model_{model_type}_data_{dataset_type}.png")

    pd.DataFrame(model_losses).to_csv('model_losses.csv')

    # Compare model performance
    for config_key, models in all_models.items():
        for model_type, model in models.items():
            for dataset_type in dataset_types:
                print(f"Config: {config_key}, Model type: {model_type}, Dataset: {dataset_type}")
                print(f"Mean: {np.mean(model_losses[(config_key, model_type, dataset_type)])}")
                print(f"Std: {np.std(model_losses[(config_key, model_type, dataset_type)])}")
                plt.hist(model_losses[(config_key, model_type, dataset_type)], bins=100)
                plt.savefig(f"hist_{config_key}_model_{model_type}_data_{dataset_type}.png")




    # Compare model performance
    means = {}
    std_devs = {}
    for key in model_losses:
        config_key, model_type, dataset_type = key
        means[key] = np.mean(model_losses[key])
        std_devs[key] = np.std(model_losses[key])
        # Generate histogram for each configuration
        plt.figure()
        plt.hist(model_losses[key], bins=10, alpha=0.75, label=f'{model_type} - {dataset_type}')
        plt.title(f"Error Distribution for {config_key}")
        plt.xlabel("Prediction Error")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig(f"hist_{config_key}_model_{model_type}_data_{dataset_type}.png")

    # Create comparative plots
    for config_key in set(k[0] for k in model_losses.keys()):
        plt.figure()
        for dataset_type in dataset_types:
            errors = [means[(config_key, model_type, dataset_type)] for model_type in ['all', 'nature', 'urban']]
            plt.bar(['all', 'nature', 'urban'], errors, alpha=0.7, label=f'{dataset_type}')
        plt.title(f'Mean Prediction Errors for {config_key}')
        plt.xlabel('Model Type')
        plt.ylabel('Mean Error')
        plt.legend(title='Dataset Type')
        plt.savefig(f'comparison_{config_key}.png')

if __name__ == '__main__':
    main()
