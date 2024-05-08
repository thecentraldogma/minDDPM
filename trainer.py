"""
Trainer code for any nn model. 
"""

import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
from utils import CfgNode as CN
import torch
from torch.utils.tensorboard import SummaryWriter
import os

def ensure_directory_exists(path):
    """
    Check if a directory exists at the specified path; if not, create it.

    Args:
    path (str): The path to the directory to check and potentially create.

    Returns:
    None
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created at {path}")
    else:
        print(f"Directory already exists at {path}")


class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()

        # device to train on
        C.device = 'auto'
        
        # dataloder parameters
        C.num_workers = 4 
        
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 100
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95) # this is meant for AdamW
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        C.eval_iters = 500
        C.checkpoint_iter_num = 500
        C.checkpoint_path = "./checkpoints/"
        C.resume = False
        return C

    def __init__(self, config, model, train_dataset, test_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.callbacks = defaultdict(list)
        self.eval_iters = config.eval_iters
        self.batch_size = config.batch_size
        self.checkpoint_iter_num = config.checkpoint_iter_num
        self.checkpoint_path = config.checkpoint_path
        self.resume = config.resume

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)



    def run(self):
        config = self.config
        model = self.model
        self.optimizer = model.configure_optimizers(config)

        # check if we are supposed to resume training from a checkpoint
        if self.resume and self.checkpoint_path is not None: 
            checkpoint = torch.load(self.checkpoint_path)
            model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.iter_num = checkpoint['iters'] + 1
            print(f"Resuming training from epoch {self.iter_num}")
        else: 
            self.iter_num = 0 
        
        
        # Create a tensorboard SummaryWriter object
        writer = SummaryWriter()

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            #sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=True,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            drop_last=True
        )

        test_loader = DataLoader(
            self.test_dataset,
            shuffle=True,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            drop_last=True  
        )

        self.iter_time = time.time()
        data_iter = iter(train_loader)
        self.train_losses = []
        self.test_losses = []
        while True:

            model.train()
            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y, t = batch # x = inputs, y = target (noise to predict), t = time step

            # forward the model
            preds, self.loss = model(x, t, y)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            #self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # Compuse loss on test dataset and train dataset, every self.eval_iters
            if self.iter_num % self.eval_iters == 0:
                model.eval()
                test_losses = []
                train_losses = []
                with torch.inference_mode():
                    for b, (x_test_batch, y_test_batch, t_test_batch) in enumerate(test_loader):
                        x_test_batch = x_test_batch.to(self.device)
                        y_test_batch = y_test_batch.to(self.device)
                        t_test_batch = t_test_batch.to(self.device)
                        test_preds, test_loss = model(x_test_batch, t_test_batch, y_test_batch)
                        test_losses.append(test_loss)
                    for b, (x_train_batch, y_train_batch, t_train_batch) in enumerate(train_loader):
                        x_train_batch = x_train_batch.to(self.device)
                        y_train_batch = y_train_batch.to(self.device)
                        t_train_batch = t_train_batch.to(self.device)
                        train_preds, train_loss = model(x_train_batch, t_train_batch, y_train_batch)
                        train_losses.append(train_loss)
                        

                mean_training_loss = torch.tensor(train_losses).mean().item()
                mean_test_loss = torch.tensor(test_losses).mean().item()
                batch_loss = self.loss.item()

                print("iter_num", self.iter_num, " train_loss:", mean_training_loss, ", test_loss: ", mean_test_loss, ", last batch loss:", batch_loss)
                self.train_losses.append([self.iter_num, mean_training_loss])
                self.test_losses.append([self.iter_num, mean_test_loss])
                writer.add_scalar("train/loss", mean_training_loss, self.iter_num)
                writer.add_scalar("validation/loss", mean_test_loss, self.iter_num)
                writer.add_scalar("batch/loss", batch_loss, self.iter_num)

            # if self.iter_num % self.checkpoint_iter_num == 0: 
            #     # Create the path if it doesn't exist
            #     ensure_directory_exists(self.checkpoint_path)

            #     # Save checkpoint after each 'checkpoint_iter_num' iterations
            #     checkpoint = {
            #         'iters': self.iter_num,
            #         'model_state': model.state_dict(),
            #         'optimizer_state': self.optimizer.state_dict()
            #     }
            #     torch.save(checkpoint, f'{self.checkpoint_path}_iters_{time.time()}_{self.iter_num}.pth')
            #     print(f'Checkpoint saved at iter {self.iter_num}')


            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break

        writer.flush()


