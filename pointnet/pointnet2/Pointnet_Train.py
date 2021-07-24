import os
from pathlib import Path
import yaml
from torch.utils.data import DataLoader, Dataset, random_split
import pytorch_lightning as pl
import torch
import numpy as np
from pointnet.pointnet2.Pointnet import PointNetEncoderDecoder
from pytorch_lightning.loggers import WandbLogger
wandb_logger = WandbLogger(name='Single Grasp Episode',project='Object Grasping using Pointnet')
checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=6, period=1,mode='min',save_weights_only=True)


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

class Data(Dataset):

    def __init__(self, x, y):
        self.pointcloud = x
        self.actions = y

    def __getitem__(self, item):
        # dict = {'pc': self.pointcloud[item], 'rel_actions': self.actions[index]}
        return (self.pointcloud[item], self.actions[item])

    def __len__(self):
        return(len(self.pointcloud))

    def __remove__(self, remove_list):
        pointcloud = np.delete(self.pointcloud, remove_list)
        actions = np.delete(self.actions, remove_list)
        return pointcloud, actions

class PrepareData(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, train_partition):
        files = [file for file in os.listdir(self.data_dir) if file.endswith(".npz")]
        pointcloud = []
        actions = []
        invalid_frames = 1
        for i in range(len(files)):
            data = np.load(self.data_dir + "/" + files[i])
            if data['point_cloud'].size != 0:
                pointcloud.append(data['point_cloud'])
                actions.append(data['rel_actions'])
            else:
                print("Point cloud for the frame {} is empty hence not considered, frame number {}".format(files[i],
                                                                                                        invalid_frames))
                invalid_frames += 1

        self.dataset = Data(pointcloud, actions)
        training_part = int(train_partition * len(self.dataset))
        while (training_part % self.batch_size != 0):
            training_part += 1
        self.train_data, self.val_data = random_split(self.dataset,
                                                      [training_part, (len(self.dataset) - training_part)])

    def train_dataloader(self):
        train_data = DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        return(train_data)

    def val_dataloader(self):
        val_data = DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        return(val_data)

    def test_dataloader(self):
        test_data = DataLoader(self.test_data, self.batch_size)
        return(test_data)

def main(cfg):
    hp = hydra_params_to_dotdict(cfg)
    train_partition = hp["training_partition"]
    load_dir = hp["load_dir"]
    save_dir = hp["save_dir"]

    Data = PrepareData(load_dir, hp["batch_size"], hp["num_workers"])
    print("Prepraing the data for training...")
    Data.setup(train_partition/100)
    print("Data is prepared and training is started...")
    Net = PointNetEncoderDecoder(hp)
    # trainer = pl.Trainer(gpus=hp["gpus"], max_epochs=hp["epochs"], logger= wandb_logger)
    trainer = pl.Trainer(gpus=hp["gpus"], max_epochs=hp["epochs"], logger=wandb_logger,
                                                                             checkpoint_callback=checkpoint_callback)
    trainer.fit(Net, Data.train_dataloader(), Data.val_dataloader())
    torch.save(Net.state_dict(), save_dir)
    wandb_logger.save()

def hydra_params_to_dotdict(hparams):
    def _to_dot_dict(cfg):
        res = {}
        for k, v in cfg.items():
            if isinstance(v, dict):
                res.update(
                    {k + "." + subk: subv for subk, subv in _to_dot_dict(v).items()}
                )
            elif isinstance(v, (str, int, float, bool)):
                res[k] = v

        return res

    return _to_dot_dict(hparams)


if __name__ == "__main__":
    with open("config/Hyperparam.yaml", "r") as yamlfile:
        cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
        print("Read successful")
    print(cfg)
    main(cfg)