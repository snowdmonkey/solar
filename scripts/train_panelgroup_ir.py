from pathlib import Path
from fcn.datasets import FCNDataset
from fcn.train import FCNTrainer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler("logfile")])


feature_folders = [Path(x) for x in ["./2017-06-20",
                                     "./2017-06-21",
                                     "./2017-08-15",
                                     "./2017-09-19"]]
label_folders = [Path(x) for x in ["./2017-06-20/annotation",
                                   "./2017-06-21/annotation",
                                   "./2017-08-15/annotation",
                                   "./2017-09-19/annotation"]]


color_map = {(0, 0, 0): 0, (0, 255, 0): 1, (255, 0, 0): 1}
dataset = FCNDataset(*zip(feature_folders, label_folders), color_map=color_map, gray_scale=True)

trainer = FCNTrainer(n_classes=2, n_channels=1)
trainer.train(dataset=dataset, n_epochs=0, train_proportion=0.8)
