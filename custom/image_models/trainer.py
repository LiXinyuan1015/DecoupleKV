import os
import time
import tqdm
import math
import torch
from torch.cuda.amp import GradScaler, autocast

class Trainer(object):
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        num_epoch,
        train_dataloader, 
        eval_dataloader,
        checkpoint_path,
        logger,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epoch = num_epoch
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.checkpoint_path = checkpoint_path
        self.logger = logger

    def train(self):
        # set up scaler
        scaler = torch.amp.GradScaler()

        for i in range(self.num_epoch):
            # reset train dataloader
            self.train_dataloader._reset()
            # start training
            self.model.train()
            
            epoch_start_time = time.time()
            num_correct, num_train, num_loss = 0, 0, 0
            loss_total, ppl_total = 0., 0.
            for batch in tqdm.tqdm(self.train_dataloader):
                input_feature, label = batch["input"], batch["label"]
                with torch.amp.autocast():
                    output = self.model(input_feature, label)
                loss, label_predict = output.loss, output.label_predict
                self.model.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.scheduler.step()

                num_train += label.size(0)
                num_loss += 1
                num_correct += torch.sum(label_predict == label).item()
                loss_total += loss.item()
                ppl_total += math.exp(loss.item())
                
                # print(loss.item())

            epoch_duration = time.time() - epoch_start_time
            self.logger.info(f"Epoch {i} finished. Time consumption: {epoch_duration}s.")
            self.logger.info(f"Totally got {num_correct} / {num_train} correct, accuracy: {num_correct / num_train}")
            self.logger.info(f"Average loss value: {loss_total / num_loss}, average perplexity: {ppl_total / num_loss}")
            # start evaluation
            # self.save_checkpoint({"epoch": i})
            # self.evaluate()

    def evaluate(self):
        # reset evaluation dataloader
        self.eval_dataloader._reset()
        # start evaluation
        self.model.eval()

        with torch.no_grad():
            eval_start_time = time.time()
            num_correct, num_train, num_loss = 0, 0, 0
            loss_total, ppl_total = 0., 0.
            for batch in tqdm.tqdm(self.eval_dataloader):
                input_feature, label = batch["input"], batch["label"]
                _, label_predict, loss = self.model(input_feature, label)
                self.model.zero_grad()

                num_train += label.size(0)
                num_loss += 1
                num_correct += torch.sum(label_predict == label).item()
                loss_total += loss.item()
                ppl_total += math.exp(loss.item())

            eval_duration = time.time() - eval_start_time
            self.logger.info(f"Evaluation finished. Time consumption: {eval_duration}s.")
            self.logger.info(f"Totally got {num_correct} / {num_train} correct, accuracy: {num_correct / num_train}")
            self.logger.info(f"Average loss value: {loss_total / num_loss}, average perplexity: {ppl_total / num_loss}")
            
    def save_checkpoint(self, info):
        checkpoint_name = "checkpoint"
        for key, value in info.items():
            checkpoint_name += "_".join(["", str(key), str(value)])
        checkpoint_name += ".bin"
        saving_path = os.path.join(self.checkpoint_path, checkpoint_name)
        checkpoint = {
            "model": self.model.state_dict(), 
            "optimizer": self.optimizer.state_dict(), 
            "scheduler": self.scheduler.state_dict()
        }
        self.logger.info(f"Saving checkpoint in {saving_path}")
        torch.save(checkpoint, saving_path)
                