import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import ConvNextModel
from pytorch_lightning import LightningModule, Trainer
import webp
from sklearn.model_selection import train_test_split
from lightly.transforms.utils import IMAGENET_NORMALIZE
import xml.etree.ElementTree as ET
import re
import math

class NaturnessDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = webp.load_image(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

class NaturenessRegressionModel(LightningModule):
    def __init__(self, backbone, freeze_backbone=False):
        super().__init__()
        self.backbone = backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.fc = nn.Linear(768, 1)
        self.criterion = nn.MSELoss()

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
        optim = torch.optim.Adam(self.parameters(), lr=0.002884)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=100)
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

def load_data(data_dir, threshold):
    image_paths = glob.glob(os.path.join(data_dir, '**', '*.webp'), recursive=True)
    xml_paths = glob.glob(os.path.join(data_dir, '**', '*.xml'), recursive=True)
    labels = []
    for xml_path in xml_paths:
        filename, bndbox_areas = parse_xml_for_bndbox_areas(xml_path)
        # Find filename in image_paths and get the index (image_path can be in subdirectories)
        # index = [i for i, path in enumerate(image_paths) if path.endswith(filename)][0]
        try:
            labels.append(calculate_natureness_score(bndbox_areas))
        except:
            labels.append(-100.) # invalid value
    labels = np.array(labels).astype(np.float32)
    image_paths = np.array(image_paths)
    # Filter out the images with invalid label
    image_paths = image_paths[labels != -100.]
    labels = labels[labels != -100.]
    print(f"Loaded {len(image_paths)} images and {len(labels)} labels")
    nature_x = image_paths[labels >= threshold]
    nature_y = labels[labels >= threshold]
    city_x = image_paths[labels < threshold]
    city_y = labels[labels < threshold]
    print(f"Size of nature dataset: {len(nature_y)}, size of city dataset: {len(city_y)}")
    return {
        "nature": {
            "data": nature_x,
            "label": nature_y,
        },
        "city": {
            "data": city_x,
            "label": city_y,
        }
    }

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
        'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, persistent_workers=True),
        'val': DataLoader(datasets['val'], batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True, persistent_workers=True),
        'test': DataLoader(datasets['test'], batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True, persistent_workers=True)
    }

def main():
    data_dir = '../data'
    batch_size = 16
    num_workers = os.cpu_count() or 1
    threshold = 0.6 # labels are between 0.25 and 0.75
    
    dataset = load_data(data_dir, threshold)
    dataloaders = {}
    
    for data_type in ["nature", "city"]:
        x_train, x_test, y_train, y_test = train_test_split(dataset[data_type]["data"], dataset[data_type]["label"],
                                                            test_size=0.2, random_state=42)
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

        dataloaders[data_type] = create_dataloaders((x_train, y_train), (x_val, y_val), (x_test, y_test), batch_size, num_workers)
        
    for model_type in ["nature", "city"]:

        model = NaturenessRegressionModel(ConvNextModel.from_pretrained("facebook/convnext-tiny-224"))

        trainer = Trainer(max_epochs=25)
        trainer.fit(model, dataloaders[model_type]['train'], dataloaders[model_type]['val'])

        # Eval
        for test_type in ["nature", "city"]:
            trainer.test(model, dataloaders[test_type]['test'])

            # Compare predicted and actual values
            model.eval()
            y_pred = []
            y_true = []
            for x, y in dataloaders[test_type]['test']:
                y_pred.extend(model(x).tolist())
                y_true.extend(y.tolist())

            # Show example images with their predicted and actual values
            for i in range(5):
                x, y = dataloaders[test_type]['test'].dataset[i]
                y_pred = model(x.unsqueeze(0)).item()
                plt.imshow(x.permute(1, 2, 0))
                plt.title(f"Predicted: {y_pred:.2f}, Actual: {y:.2f}")
                plt.show()

            # Save 5 pictures with their predicted and actual values
            for i in range(5):
                x, y = dataloaders[test_type]['test'].dataset[i]
                y_pred = model(x.unsqueeze(0)).item()
                plt.imshow(x.permute(1, 2, 0))
                plt.title(f"Predicted: {y_pred:.2f}, Actual: {y:.2f}")
                plt.savefig(f"example_{test_type}_on_{model_type}model_{i}.png")

        # Save the model
        torch.save(model.state_dict(), f"{model_type}ness_model.pth")

if __name__ == '__main__':
    main()