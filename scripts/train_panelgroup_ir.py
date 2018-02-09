import logging
from pathlib import Path

from torchvision import transforms

from fcn.datasets import FCNDataset, RandomScale, RandomCrop, ToTensor
from fcn.train import FCNTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(), logging.FileHandler("logfile")])

feature_folders = [Path(x) for x in [
    "./2017-06-20",
    "./2017-06-21",
    "./2017-08-15",
    "./2017-09-19"]]
label_folders = [Path(x) for x in [
    "./2017-06-20/annotation",
    "./2017-06-21/annotation",
    "./2017-08-15/annotation",
    "./2017-09-19/annotation"]]

color_map = {(0, 0, 0): 0, (0, 255, 0): 1, (255, 0, 0): 2}
dataset = FCNDataset(*zip(feature_folders, label_folders), color_map=color_map, gray_scale=True,
                     transform=transforms.Compose([RandomScale(), RandomCrop(height=256, width=336), ToTensor()]))
                     # transform=transforms.Compose([ToTensor()]))

trainer = FCNTrainer(n_classes=2, n_channels=1)
trainer.set_dataset(dataset, 0.8, batch_size=4)
trainer.train(n_epochs=300)
