import torch
import math
import os
import time
import copy
import numpy as np
import csv
from lib.logger import get_logger
from lib.metrics import All_Metrics

class Trainer:
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.lr_scheduler = lr_scheduler
        
        self.train_per_epoch = len(train_loader) if train_loader else 0  # ✅ Ensures it’s never undefined



        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir, exist_ok=True)  

        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info(f"Experiment log path in: {args.log_dir}")

        self.metrics_path = os.path.join(self.args.log_dir, "training_metrics.csv")
        # # Logging setup
        # if not os.path.isdir(args.log_dir) and not args.debug:
        #     os.makedirs(args.log_dir, exist_ok=True)
        # self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        # self.logger.info(f"Experiment log path in: {args.log_dir}")



    def val_epoch(self, epoch, val_dataloader):
        start_time = time.time()
        self.model.eval()
        total_val_loss = 0
        y_pred = []
        y_true = []
    
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                data = data[..., :self.args.input_dim]
                label = target[..., :self.args.output_dim]
    
                #  residual decomposition output
                output = self.model(data)
                #output = output1 + output2  # Sum major trend and residual
    
                y_true.append(label)
                y_pred.append(output)
    
                if self.args.real_value:
                    label = self.scaler.inverse_transform(label)
                loss = self.loss(output, label)
                if not torch.isnan(loss).any():
                    total_val_loss += loss.item()
    
        val_loss = total_val_loss / len(val_dataloader)
        y_true = self.scaler.inverse_transform(torch.cat(y_true, dim=0))
        y_pred = torch.cat(y_pred, dim=0) if self.args.real_value else self.scaler.inverse_transform(torch.cat(y_pred, dim=0))
    
        mae, rmse, mape, _, _ = All_Metrics(y_pred, y_true, self.args.mae_thresh, self.args.mape_thresh)
        epoch_duration = time.time() - start_time
        self.val_mae = mae
        self.val_rmse = rmse
        self.val_mape = mape
        self.val_time = epoch_duration
        self.logger.info(f"Validation Epoch {epoch}: average Loss: {val_loss:.6f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}, Time: {epoch_duration:.2f} seconds")
        return val_loss

    def train_epoch(self, epoch):
        start_time = time.time()
        self.model.train()
        total_loss = 0
        y_pred = []
        y_true = []
    
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data[..., :self.args.input_dim]
            label = target[..., :self.args.output_dim]
            self.optimizer.zero_grad()
    
            if self.args.teacher_forcing:
                global_step = (epoch - 1) * self.train_per_epoch + batch_idx
                teacher_forcing_ratio = self._compute_sampling_threshold(global_step, self.args.tf_decay_steps)
            else:
                teacher_forcing_ratio = 1.0
    
            #  residual decomposition output
            output = self.model(data)
            #output = output1 + output2  # Sum major trend and residual
    
            # Debugging residual values
            #print(f"[DEBUG] output1 mean: {output1.mean().item()}, output2 mean: {output2.mean().item()}")
    
            y_true.append(label)
            y_pred.append(output)
    
            if self.args.real_value:
                label = self.scaler.inverse_transform(label)
            loss = self.loss(output, label)
            loss.backward()
    
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()
    
            if batch_idx % self.args.log_step == 0:
                self.logger.info(f"Train Epoch {epoch}: {batch_idx}/{self.train_per_epoch} Loss: {loss.item():.6f}")
    
        train_epoch_loss = total_loss / self.train_per_epoch
        y_true = self.scaler.inverse_transform(torch.cat(y_true, dim=0))
        y_pred = torch.cat(y_pred, dim=0) if self.args.real_value else self.scaler.inverse_transform(torch.cat(y_pred, dim=0))
    
        mae, rmse, mape, _, _ = All_Metrics(y_pred, y_true, self.args.mae_thresh, self.args.mape_thresh)
        epoch_duration = time.time() - start_time
        self.train_mae = mae
        self.train_rmse = rmse
        self.train_mape = mape
        self.train_time = epoch_duration
        self.logger.info(f"Train Epoch {epoch}: averaged Loss: {train_epoch_loss:.6f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}, tf_ratio: {teacher_forcing_ratio:.6f}, Time: {epoch_duration:.2f} seconds")
    
        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss

    def log_gpu_memory(self, epoch):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # Convert to GB
            self.logger.info(f"Epoch {epoch}: GPU Memory Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        metrics_path = os.path.join(self.args.log_dir, "training_metrics.csv")
        headers_written = False

        for epoch in range(1, self.args.epochs + 1):
            epoch_start_time = time.time()

            # Unfreeze embeddings at epoch 20
            if epoch == 1 and hasattr(self.model, 'set_embedding_trainable'):
                self.model.set_embedding_trainable(True)
                self.logger.info("Unfrozen node embeddings after 1 epochs")

            train_epoch_loss = self.train_epoch(epoch)
            val_dataloader = self.val_loader if self.val_loader is not None else self.test_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)

            # Log GPU memory usage
            self.log_gpu_memory(epoch)

            # Prepare and save metrics
            metrics = {
                "epoch": epoch,
                "train": {
                    "loss": train_epoch_loss,
                    "mae": self.train_mae,
                    "rmse": self.train_rmse,
                    "mape": self.train_mape,
                    "time": self.train_time
                },
                "validation": {
                    "loss": val_epoch_loss,
                    "mae": self.val_mae,
                    "rmse": self.val_rmse,
                    "mape": self.val_mape,
                    "time": self.val_time
                }
            }

            flattened_metrics = {
                "epoch": metrics["epoch"],
                "train_loss": metrics["train"]["loss"],
                "train_mae": metrics["train"]["mae"],
                "train_rmse": metrics["train"]["rmse"],
                "train_mape": metrics["train"]["mape"],
                "train_time": metrics["train"]["time"],
                "val_loss": metrics["validation"]["loss"],
                "val_mae": metrics["validation"]["mae"],
                "val_rmse": metrics["validation"]["rmse"],
                "val_mape": metrics["validation"]["mape"],
                "val_time": metrics["validation"]["time"]
            }

            with open(metrics_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=flattened_metrics.keys())
                if not headers_written:
                    writer.writeheader()
                    headers_written = True
                writer.writerow(flattened_metrics)

            # Clear memory to avoid GPU issues
            del metrics, flattened_metrics
            torch.cuda.empty_cache()

            # Log epoch time
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            self.logger.info(f"Epoch {epoch} completed in {epoch_duration:.2f} seconds")

            # Save best model
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_model = copy.deepcopy(self.model.state_dict())
                self.logger.info("********** Current best model saved!")
            else:
                not_improved_count += 1

            # Early stopping
            if self.args.early_stop and not_improved_count >= self.args.early_stop_patience:
                self.logger.info(f"Validation performance didn't improve for {self.args.early_stop_patience} epochs. Stopping...")
                break

        self.logger.info(f"Metrics saved to {metrics_path}")

        # Save the best model
        if best_model:
            if not self.args.debug:
                torch.save(best_model, self.best_path)
                self.logger.info(f"Best model saved at {self.best_path}")

        # Load the best model for testing
        self.model.load_state_dict(best_model)
        self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)
        
    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info(f"Saving current best model to {self.best_path}")

    @staticmethod
    def test(model, args, data_loader, scaler, logger, path=None):
        if path:
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['state_dict'])
            model.to(args.device)
        model.eval()

        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data[..., :args.input_dim]
                label = target[..., :args.output_dim]


                output = model(data)
                if output is None or output.numel() == 0:
                    #print(f"[ERROR] Model returned an empty tensor at batch {batch_idx}")
                    continue  # رد کردن این batch

                y_true.append(label)
                y_pred.append(output)

        if len(y_true) == 0 or len(y_pred) == 0:
            #print("[ERROR] No valid predictions collected in test()!")
            return float('inf')

        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        y_pred = torch.cat(y_pred, dim=0) if args.real_value else scaler.inverse_transform(torch.cat(y_pred, dim=0))

        np.save(f"./{args.dataset}_true.npy", y_true.cpu().numpy())
        np.save(f"./{args.dataset}_pred.npy", y_pred.cpu().numpy())

        for t in range(y_true.shape[1]):
            mae, rmse, mape, _, _ = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...], args.mae_thresh, args.mape_thresh)
            logger.info(f"Horizon {t + 1:02d}: MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.4f}%")

        mae, rmse, mape, _, _ = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        logger.info(f"Average Horizon: MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.4f}%")

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        return k / (k + math.exp(global_step / k))

