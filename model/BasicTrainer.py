import torch
import math
import os
import time
import copy
import numpy as np
import importlib.util
import csv
from lib.logger import get_logger
from lib.metrics import All_Metrics
from model.SGCRN import SGCRN
import matplotlib.pyplot as plt  # noqa: WPS433 - optional visualisation


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

        # Safe guard for empty loaders
        self.train_per_epoch = len(train_loader) if train_loader else 0

        # Logging dirs
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir, exist_ok=True)

        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info(f"Experiment log path in: {args.log_dir}")

        self.metrics_path = os.path.join(self.args.log_dir, "training_metrics.csv")
        self.best_path = os.path.join(self.args.log_dir, "best_model.pth")

        # internal state for CSV header handling
        self._metrics_header_written = False

    def val_epoch(self, epoch, val_dataloader):
        start_time = time.time()
        self.model.eval()
        total_val_loss = 0.0
        y_pred = []
        y_true = []

        if not val_dataloader or len(val_dataloader) == 0:
            self.logger.warning("Validation dataloader is empty; skipping validation epoch.")
            self.val_mae = self.val_rmse = self.val_mape = float("nan")
            self.val_time = 0.0
            return float("inf")

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                data = data[..., :self.args.input_dim]
                label = target[..., :self.args.output_dim]

                output = self.model(data)

                y_true.append(label)
                y_pred.append(output)

                label_for_loss = label
                if self.args.real_value:
                    label_for_loss = self.scaler.inverse_transform(label)
                loss = self.loss(output, label_for_loss)
                if not torch.isnan(loss).any():
                    total_val_loss += loss.item()

        val_loss = total_val_loss / max(1, len(val_dataloader))

        # Metrics in real space
        y_true = self.scaler.inverse_transform(torch.cat(y_true, dim=0))
        y_pred_cat = torch.cat(y_pred, dim=0)
        y_pred_real = y_pred_cat if self.args.real_value else self.scaler.inverse_transform(y_pred_cat)

        mae, rmse, mape, _, _ = All_Metrics(y_pred_real, y_true, self.args.mae_thresh, self.args.mape_thresh)
        epoch_duration = time.time() - start_time
        self.val_mae = mae
        self.val_rmse = rmse
        self.val_mape = mape
        self.val_time = epoch_duration
        self.logger.info(
            f"Validation Epoch {epoch}: average Loss: {val_loss:.6f}, "
            f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}, "
            f"Time: {epoch_duration:.2f} seconds"
        )
        return val_loss

    def train_epoch(self, epoch):
        start_time = time.time()
        self.model.train()
        total_loss = 0.0
        y_pred = []
        y_true = []

        if self.train_per_epoch == 0:
            self.logger.warning("Train dataloader is empty; skipping train epoch.")
            self.train_mae = self.train_rmse = self.train_mape = float("nan")
            self.train_time = 0.0
            return 0.0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data[..., :self.args.input_dim]
            label = target[..., :self.args.output_dim]
            self.optimizer.zero_grad()

            if getattr(self.args, "teacher_forcing", False):
                global_step = (epoch - 1) * self.train_per_epoch + batch_idx
                teacher_forcing_ratio = self._compute_sampling_threshold(global_step, self.args.tf_decay_steps)
            else:
                teacher_forcing_ratio = 1.0

            # Forward
            output = self.model(data)

            # Collect for metrics
            y_true.append(label)
            y_pred.append(output)

            # Loss (optionally compute against inverse-transformed labels)
            label_for_loss = label
            if self.args.real_value:
                label_for_loss = self.scaler.inverse_transform(label)

            loss = self.loss(output, label_for_loss)
            loss.backward()

            if getattr(self.args, "grad_norm", False):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

            self.optimizer.step()
            total_loss += loss.item()

            if batch_idx % self.args.log_step == 0:
                self.logger.info(
                    f"Train Epoch {epoch}: {batch_idx}/{self.train_per_epoch} "
                    f"Loss: {loss.item():.6f}"
                )

        train_epoch_loss = total_loss / max(1, self.train_per_epoch)

        # Metrics in real space
        y_true = self.scaler.inverse_transform(torch.cat(y_true, dim=0))
        y_pred_cat = torch.cat(y_pred, dim=0)
        y_pred_real = y_pred_cat if self.args.real_value else self.scaler.inverse_transform(y_pred_cat)

        mae, rmse, mape, _, _ = All_Metrics(y_pred_real, y_true, self.args.mae_thresh, self.args.mape_thresh)
        epoch_duration = time.time() - start_time
        self.train_mae = mae
        self.train_rmse = rmse
        self.train_mape = mape
        self.train_time = epoch_duration

        self.logger.info(
            f"Train Epoch {epoch}: averaged Loss: {train_epoch_loss:.6f}, "
            f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}, "
            f"tf_ratio: {teacher_forcing_ratio:.6f}, Time: {epoch_duration:.2f} seconds"
        )

        if getattr(self.args, "lr_decay", False) and self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return train_epoch_loss

    def log_gpu_memory(self, epoch):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
            self.logger.info(
                f"Epoch {epoch}: GPU Memory Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB"
            )

    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0

        for epoch in range(1, self.args.epochs + 1):
            epoch_start_time = time.time()

            # Unfreeze embeddings at epoch 1
            if epoch == 1 and hasattr(self.model, 'set_embedding_trainable'):
                self.model.set_embedding_trainable(True)
                self.logger.info("Unfrozen node embeddings after 1 epochs")

            train_epoch_loss = self.train_epoch(epoch)
            val_dataloader = self.val_loader if self.val_loader is not None else self.test_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)

            # Log GPU memory usage
            self.log_gpu_memory(epoch)

            # Prepare and save flattened metrics
            flattened_metrics = {
                "epoch": epoch,
                "train_loss": train_epoch_loss,
                "train_mae": getattr(self, "train_mae", float("nan")),
                "train_rmse": getattr(self, "train_rmse", float("nan")),
                "train_mape": getattr(self, "train_mape", float("nan")),
                "train_time": getattr(self, "train_time", float("nan")),
                "val_loss": val_epoch_loss,
                "val_mae": getattr(self, "val_mae", float("nan")),
                "val_rmse": getattr(self, "val_rmse", float("nan")),
                "val_mape": getattr(self, "val_mape", float("nan")),
                "val_time": getattr(self, "val_time", float("nan")),
            }

            # Append metrics to CSV (write header once)
            with open(self.metrics_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(flattened_metrics.keys()))
                if not self._metrics_header_written or (os.path.getsize(self.metrics_path) == 0):
                    writer.writeheader()
                    self._metrics_header_written = True
                writer.writerow(flattened_metrics)

            # Clear memory
            del flattened_metrics
            torch.cuda.empty_cache()

            # Log epoch time
            epoch_duration = time.time() - epoch_start_time
            self.logger.info(f"Epoch {epoch} completed in {epoch_duration:.2f} seconds")

            # Save best model (by val loss)
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_model = copy.deepcopy(self.model.state_dict())
                self.logger.info("********** Current best model saved!")
            else:
                not_improved_count += 1

            # Early stopping
            if getattr(self.args, "early_stop", False) and \
               not_improved_count >= self.args.early_stop_patience:
                self.logger.info(
                    f"Validation performance didn't improve for "
                    f"{self.args.early_stop_patience} epochs. Stopping..."
                )
                break

        self.logger.info(f"Metrics saved to {self.metrics_path}")

        # Save and evaluate best model
        if best_model:
            if not self.args.debug:
                torch.save(best_model, self.best_path)
                self.logger.info(f"Best model saved at {self.best_path}")

            self.model.load_state_dict(best_model)
            self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)
            self._export_learned_graph()
        else:
            self.logger.warning("No best model was captured during training; skipping export.")

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
                    continue

                y_true.append(label)
                y_pred.append(output)

        if len(y_true) == 0 or len(y_pred) == 0:
            logger.error("No valid predictions collected in test()!")
            return float('inf')

        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        y_pred_cat = torch.cat(y_pred, dim=0)
        y_pred_real = y_pred_cat if args.real_value else scaler.inverse_transform(y_pred_cat)

        np.save(f"./{args.dataset}_true.npy", y_true.cpu().numpy())
        np.save(f"./{args.dataset}_pred.npy", y_pred_real.cpu().numpy())

        for t in range(y_true.shape[1]):
            mae, rmse, mape, _, _ = All_Metrics(
                y_pred_real[:, t, ...], y_true[:, t, ...], args.mae_thresh, args.mape_thresh
            )
            logger.info(f"Horizon {t + 1:02d}: MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.4f}%")

        mae, rmse, mape, _, _ = All_Metrics(
            y_pred_real, y_true, args.mae_thresh, args.mape_thresh
        )
        logger.info(f"Average Horizon: MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.4f}%")

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        return k / (k + math.exp(global_step / k))

    def _export_learned_graph(self):
        """
        Export the learned adaptive graph as .npy and a heatmap PNG.
        Requires model to implement .get_adaptive_adj().
        """
        if not hasattr(self.model, "get_adaptive_adj"):
            self.logger.warning("Model does not expose an adaptive graph; skipping export.")
            return

        # Save matrix
        with torch.no_grad():
            learned_adj = self.model.get_adaptive_adj().detach().cpu().numpy()

        graph_path = os.path.join(self.args.log_dir, "adaptive_graph.npy")
        np.save(graph_path, learned_adj)
        self.logger.info(f"Saved learned adaptive graph to {graph_path}")

        # Save heatmap (if matplotlib is available)
        if importlib.util.find_spec("matplotlib") is None:
            self.logger.warning("matplotlib not available; skipping heatmap export.")
            return

        heatmap_path = os.path.join(self.args.log_dir, "adaptive_graph_heatmap.png")
        plt.figure(figsize=(8, 6))
        im = plt.imshow(learned_adj, cmap="viridis")
        plt.title("Learned Adaptive Graph")
        plt.xlabel("Target Node")
        plt.ylabel("Source Node")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=300)
        plt.close()
        self.logger.info(f"Saved adaptive graph heatmap to {heatmap_path}")
