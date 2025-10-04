from typing import Optional, Tuple

import os
import random
import numpy as np
import logging
from tqdm import tqdm
import warnings
import torch
from torch.utils.data import Subset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
from torch_scatter import scatter
from torch_geometric.loader import DataLoader

from ..hparams import DataArguments, ModelArguments, TrainingArguments
from ..datasets import PCQM4MV2_XYZ
from ..models.models_3d import LNNP
from ..utils import make_splits


class DataModule:
    def __init__(self, data_args, model_args, training_args):
        self.data_args = data_args
        self.model_args = model_args
        self.training_args = training_args
        
        self._mean = None
        self._std = None
        self.dataset = None
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def prepare_data(self):
        print("Preparing data...")
        # --- 1. 加载数据集和定义转换 (Transform) ---
        if self.data_args.position_noise_scale > 0.:
            def transform(data):
                noise = torch.randn_like(data.pos) * self.data_args.position_noise_scale
                data.pos_target = noise
                data.pos = data.pos + noise
                return data
        else:
            transform = None

        dataset_factory = lambda t: PCQM4MV2_XYZ(root=self.data_args.dataset_root, dataset_arg=self.data_args.dataset_arg, transform=t)
        
        # 带有噪声的数据集（用于训练）
        self.dataset_maybe_noisy = dataset_factory(transform)
        # 干净的数据集（用于验证/测试）
        self.dataset = dataset_factory(None)

        # --- 2. 划分数据集索引 ---
        idx_train, idx_val, idx_test = make_splits(
            dataset_len=len(self.dataset),
            train_size=self.training_args.train_size,
            val_size=self.training_args.val_size,
            test_size=self.training_args.test_size,
            seed=self.training_args.seed,
            filename=os.path.join(self.training_args.log_dir, "splits.npz"),
            order=None,
        )
        print(f"Split sizes: train {len(idx_train)}, val {len(idx_val)}, test {len(idx_test)}")

        # --- 3. 创建 Subset ---
        self.train_dataset = Subset(self.dataset_maybe_noisy, idx_train)

        if self.data_args.denoising_only:
            # 如果只进行去噪任务，验证和测试集也使用带噪声的数据
            self.val_dataset = Subset(self.dataset_maybe_noisy, idx_val)
            self.test_dataset = Subset(self.dataset_maybe_noisy, idx_test)
        else:
            self.val_dataset = Subset(self.dataset, idx_val)
            self.test_dataset = Subset(self.dataset, idx_test)

        # --- 4. 如果需要，进行数据标准化 ---
        if self.model_args.standardize:
            self._standardize()
        print("Data preparation complete.")

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloaders()["train"]
    
    def val_dataloader(self) -> DataLoader:
        return self.get_dataloaders()["val"]
    
    def test_dataloader(self) -> DataLoader:
        return self.get_dataloaders()["test"]

    def get_dataloaders(self) -> dict:
        if not all([self.train_dataset, self.val_dataset, self.test_dataset]):
            raise RuntimeError("Data has not been prepared. Please call `prepare_data()` first.")

        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.training_args.train_batch_size,
            shuffle=True,
            num_workers=self.training_args.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.training_args.val_batch_size,
            shuffle=False,
            num_workers=self.training_args.num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.training_args.test_batch_size,
            shuffle=False,
            num_workers=self.training_args.num_workers,
            pin_memory=True,
        )
        
        return {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader
        }

    def _standardize(self) -> None:
        print("Computing mean and std for standardization...")
        
        # 临时的 DataLoader 用于计算
        temp_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.training_args.val_batch_size, # 使用验证集的 batch size 即可
            shuffle=False, 
            num_workers=self.training_args.num_workers
        )

        def get_energy(batch, atomref):
            if atomref is None:
                return batch.y.clone()
            atomref_energy = scatter(atomref[batch.z], batch.batch, dim=0)
            return (batch.y.squeeze() - atomref_energy.squeeze()).clone()

        try:
            # 使用列表推导和 torch.cat 来高效处理
            ys = torch.cat([get_energy(batch, None) for batch in tqdm(temp_loader, desc="Standardizing")])
        except Exception as e:
            warnings.warn(f"Failed to compute mean and std. Error: {e}")
            return

        self._mean = ys.mean(dim=0)
        self._std = ys.std(dim=0)
        print(f"Mean: {self._mean}, Std: {self._std}")

    # --- Properties to access computed values ---
    @property
    def atomref(self) -> torch.Tensor | None:
        if hasattr(self.dataset, "get_atomref"):
            return self.dataset.get_atomref()
        return None

    @property
    def mean(self) -> torch.Tensor | None:
        return self._mean

    @property
    def std(self) -> torch.Tensor | None:
        return self._std


class FradPretrainer:
    '''Class to handle the pretraining of the Frad model.'''

    def __init__(self, data_args: DataArguments, model_args: ModelArguments, training_args: TrainingArguments):
        '''
        Args:
            data_args (DataArguments): Arguments related to the dataset.
            model_args (ModelArguments): Arguments related to the model.
            training_args (TrainingArguments): Arguments related to training.
        '''
        self.data_args = data_args
        self.model_args = model_args
        self.training_args = training_args
        
        self.data = DataModule(self.data_args, self.model_args, self.training_args)
        self.data.prepare_data()
        self.train_dataloader = self.data.train_dataloader()
        self.val_dataloader = self.data.val_dataloader()
        self.test_dataloader = self.data.test_dataloader()

        # Initialize module
        self.model = LNNP(self.data_args, self.model_args, self.training_args, prior_model=None, mean=self.data.mean, std=self.data.std)

        # --- 初始化 ---
        self.device = torch.device(self.training_args.device)
        self.model.to(self.device)
        
        # 创建日志和模型保存目录
        os.makedirs(self.training_args.log_dir, exist_ok=True)
        
        # --- 配置优化器和学习率调度器 (与之前相同) ---
        self.optimizer = AdamW(self.model.parameters(), lr=self.training_args.lr, weight_decay=self.training_args.weight_decay)
        self.scheduler = None
        self.scheduler = CosineAnnealingLR(self.optimizer, self.training_args.lr_cosine_length)
        
        # Ensure reproducibility
        self.seed = self.training_args.seed
        self._set_seed(self.seed)

        # Setup logging
        self._setup_logging()

        # --- 配置日志记录器 (Loggers) ---
        self.writer = SummaryWriter(log_dir=os.path.join(self.training_args.log_dir, "tensorboard"))
        self.console_logger.info(
            f"Initialized MolformerPretrainer on device: {self.device}"
        )

        # --- 配置混合精度 (Mixed Precision) ---
        self.use_amp = self.training_args.fp16 and self.device.type == 'cuda'
        self.scaler = GradScaler('cuda', enabled=self.use_amp)

        # --- 初始化用于回调功能的变量 (Callbacks) ---
        self.patience_counter = 0
        self.best_model_dir = os.path.join(self.training_args.log_dir, "best_model")

    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _setup_logging(self):
        # Create loggers
        self.console_logger = logging.getLogger("ConsoleLogger")
        self.file_logger = logging.getLogger("FileLogger")
        self.console_logger.setLevel(logging.INFO)
        self.file_logger.setLevel(logging.INFO)

        # Create handlers
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(
            os.path.join(self.training_args.log_dir, "pretrain.log")
        )

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add handlers to loggers
        self.console_logger.addHandler(console_handler)
        self.console_logger.addHandler(file_handler)
        self.file_logger.addHandler(file_handler)

    def save_checkpoint(
        self,
        output_dir: str,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        epoch: int,
        global_step: int,
        best_val_loss: float,
    ):
        os.makedirs(output_dir, exist_ok=True)

        # Save model
        torch.save(
            self.model.state_dict(), os.path.join(output_dir, "model.pt")
        )

        # Save optimizer and scheduler
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

        # Save training state
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "best_val_loss": best_val_loss,
                "seed": self.seed,
            },
            os.path.join(output_dir, "training_state.pt"),
        )

    def load_checkpoint(
        self,
        checkpoint_dir: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,
    ) -> Tuple[int, int, float]:
        # Load model
        self.model.load_state_dict(
            torch.load(
                os.path.join(checkpoint_dir, "model.pt"), map_location=self.device
            )
        )

        # Load optimizer
        if optimizer and os.path.exists(os.path.join(checkpoint_dir, "optimizer.pt")):
            optimizer.load_state_dict(
                torch.load(
                    os.path.join(checkpoint_dir, "optimizer.pt"),
                    map_location=self.device,
                )
            )

        # Load scheduler
        if scheduler and os.path.exists(os.path.join(checkpoint_dir, "scheduler.pt")):
            scheduler.load_state_dict(
                torch.load(
                    os.path.join(checkpoint_dir, "scheduler.pt"),
                    map_location=self.device,
                )
            )

        # Load training state
        training_state = torch.load(
            os.path.join(checkpoint_dir, "training_state.pt"), map_location=self.device
        )

        # Return training progress
        return (
            training_state["global_step"],
            training_state["epoch"],
            training_state["best_val_loss"],
        )

    def train(self, resume_from_checkpoint: Optional[str] = None):
        # --- 训练循环 ---
        global_step = 0
        training_finished = False
        best_val_loss = float('inf')

        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            global_step, start_epoch, best_val_loss = self.load_checkpoint(
                resume_from_checkpoint, self.optimizer, self.scheduler
            )
            self.console_logger.info(
                f"Resuming from checkpoint {resume_from_checkpoint}"
            )
            self.console_logger.info(
                f"Resuming from epoch {start_epoch}, step {global_step}"
            )

        # Log training parameters
        self.console_logger.info("***** Starting training *****")
        self.console_logger.info(f"Num Epochs = {self.training_args.num_epochs}")
        self.console_logger.info(f"Batch size = {self.train_dataloader.batch_size}")

        for epoch in tqdm(range(1, self.training_args.num_epochs + 1), desc="Training epoch"):
            self.model.train()
            train_epoch_loss = 0.0

            epoch_iterator = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch}/{self.training_args.num_epochs}",
                total=len(self.train_dataloader),
                leave=False,
            )

            for step, batch in enumerate(epoch_iterator):
                batch = batch.to(self.device)
                
                # 学习率预热
                if global_step < self.training_args.lr_warmup_steps:
                    lr_scale = min(1.0, float(global_step + 1) / self.training_args.lr_warmup_steps)
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = lr_scale * self.training_args.lr
                
                # 使用 autocast 实现混合精度
                with autocast('cuda', enabled=self.use_amp):
                    losses = self.model.compute_loss(batch, stage="train")
                    loss = losses["total_loss"]
                
                # 使用 GradScaler 进行反向传播
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                train_epoch_loss += loss.item()
                
                self.scheduler.step()

                # Update progress bar
                epoch_iterator.set_postfix(loss=f"{loss.item():.4f}")
                
                # 日志记录 (每步)
                if global_step % 10 == 0: # 减少打印频率
                    self.file_logger.info(
                        f"Epoch {epoch}, Step {step}/{len(self.train_dataloader)}, "
                        f"Train Loss: {loss.item():.4f}"
                    )
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.writer.add_scalar('Loss/train_step', loss.item(), global_step)
                    self.writer.add_scalar('LearningRate/step', current_lr, global_step)

                global_step += 1
                # 检查是否达到最大步数
                if self.training_args.num_steps > 0 and global_step >= self.training_args.num_steps:
                    training_finished = True
                    break
            
            if training_finished:
                break

            avg_train_loss = train_epoch_loss / len(self.train_dataloader)

            # --- 验证阶段 ---
            self.model.eval()
            val_epoch_loss = 0.0
            eval_iterator = tqdm(self.val_dataloader, desc="Evaluating", leave=False)
            with torch.no_grad():
                for step, batch in enumerate(eval_iterator):
                    batch = batch.to(self.device)
                    losses = self.model.compute_loss(batch, stage="val")
                    loss = losses["total_loss"]
                    # 日志记录（每步）
                    if global_step % 10 == 0: # 减少打印频率
                        self.file_logger.info(
                            f"Epoch {epoch}, Step {step}/{len(self.val_dataloader)}, "
                            f"Validation Loss: {loss.item():.4f}"
                        )
                    val_epoch_loss += loss.item()
            
            avg_val_loss = val_epoch_loss / len(self.val_dataloader)
            
            self.file_logger.info(f"Epoch {epoch}/{self.training_args.num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            # --- 日志记录 (每个 Epoch) ---
            self.writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
            self.writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
            self.writer.add_scalars('Loss/train_val_epoch', {'train': avg_train_loss, 'val': avg_val_loss}, epoch)

            # save checkpoint every epoch
            self.save_checkpoint(
                os.path.join(self.training_args.log_dir, f"checkpoint-{global_step}"),
                self.optimizer,
                self.scheduler,
                epoch,
                global_step,
                avg_val_loss,
            )
            self.file_logger.info(
                f"Saved checkpoint to {os.path.join(self.training_args.log_dir, f"checkpoint-{global_step}")}"
            )

            # --- 回调逻辑: ModelCheckpoint 和 EarlyStopping ---
            if avg_val_loss < best_val_loss:
                self.file_logger.info(
                    f"Validation loss improved ({best_val_loss:.4f} -> {avg_val_loss:.4f}). Saving model..."
                )
                best_val_loss = avg_val_loss
                patience_counter = 0 # 重置耐心计数器
                # 保存最佳模型 (save_top_k=1)
                self.save_checkpoint(
                    os.path.join(self.training_args.log_dir, "best_model"),
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    global_step,
                    best_val_loss,
                )
                self.file_logger.info(
                    f"New best model saved with validation loss: {best_val_loss:.4f}"
                )
            else:
                patience_counter += 1
                self.file_logger.info(f"Validation loss did not improve. Patience: {patience_counter}/{self.training_args.early_stopping_patience}")
            
            if patience_counter >= self.training_args.early_stopping_patience:
                self.file_logger.info("Early stopping triggered. Training finished.")
                break
        
        # Save final model
        self.save_checkpoint(
            os.path.join(self.training_args.log_dir, "final_model"),
            self.optimizer,
            self.scheduler,
            epoch,
            global_step,
            best_val_loss,
        )
        self.console_logger.info(
            f"Training loop completed. Final model saved to {os.path.join(self.training_args.log_dir, "final_model")}"
        )

        self.writer.close()

        # --- 测试阶段 (trainer.test) ---
        self.console_logger.info("Running test set evaluation with the best model...")
        if os.path.exists(self.best_model_dir):
            self.load_checkpoint(self.best_model_dir, self.optimizer, self.scheduler)
            self.model.eval()
            test_epoch_loss = 0.0
            test_iterator = tqdm(self.test_dataloader, desc="Testing", leave=False)
            with torch.no_grad():
                for batch in test_iterator:
                    batch = batch.to(self.device)
                    losses = self.model.compute_loss(batch, stage="test")
                    test_epoch_loss += losses["total_loss"].item()
            
            avg_test_loss = test_epoch_loss / len(self.test_dataloader)
            self.console_logger.info(f"Test Loss: {avg_test_loss:.4f}")
        else:
            self.console_logger.info("Could not find best model checkpoint to run test.")

