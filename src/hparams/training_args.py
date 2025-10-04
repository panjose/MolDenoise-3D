import os

from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class TrainingArguments:
    '''Training arguments for the training script.'''

    train_batch_size: int = field(
        default=32,
        metadata={"help": "Batch size for training."},
    )
    val_batch_size: int = field(
        default=32,
        metadata={"help": "Batch size for validation."},
    )
    test_batch_size: int = field(
        default=32,
        metadata={"help": "Batch size for testing."},
    )
    num_epochs: int = field(
        default=10,
        metadata={"help": "Total number of training epochs to perform."},
    )
    num_steps: int = field(
        default=None,               
        metadata={"help": "Total number of training steps to perform."},
    )

    lr: float = field(
        default=1e-4,
        metadata={"help": "Initial learning rate."},
    )
    lr_warmup_steps: int = field(
        default=0,
        metadata={"help": "How many steps to warm-up over. Defaults to 0 for no warm-up"},
    )
    lr_cosine_length: int = field(
        default=400000,
        metadata={"help": "Cosine length if lr_schedule is cosine."},
    )

    early_stopping_patience: int = field(
        default=30,
        metadata={"help": "Stop training after this many epochs without improvement"},
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay strength"},
    )
    ema_alpha_y: float = field(
        default=1.0,
        metadata={"help": "The amount of influence of new losses on the exponential moving average of y"},
    )
    ema_alpha_dy: float = field(
        default=1.0,
        metadata={"help": "The amount of influence of new losses on the exponential moving average of dy"},
    )

    device: str = field(
        default="cuda:0",
        metadata={"help": "Device to use (cuda or cpu)"},
    )
    num_nodes: int = field(
        default=1,
        metadata={"help": "Number of nodes"},
    )
    fp16: bool = field(
        default=False,
        metadata={
            "help": "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"
        },
    )
    log_dir: Optional[str] = field(
        default="/logs",
        metadata={"help": "log file"},
    )
    train_size: Optional[float] = field(
        default=None,
        metadata={"help": "Percentage/number of samples in training set (None to use all remaining samples)"},
    )
    val_size: Optional[float] = field(
        default=0.05,
        metadata={"help": "Percentage/number of samples in validation set (None to use all remaining samples)"},
    )
    test_size: Optional[float] = field(
        default=0.1,    
        metadata={"help": "Percentage/number of samples in test set (None to use all remaining samples)"},       
    )
    seed: int = field(
        default=1,
        metadata={"help": "random seed (default: 1)"},
    )
    num_workers: int = field(
        default=4,
        metadata={"help": "Number of workers for data prefetch"},
    )
    job_id: Optional[str] = field(
        default="auto",
        metadata={"help": "Job ID. If auto, pick the next available numeric job id."},
    )
    pretrained_model: Optional[str] = field(
        default=None,
        metadata={"help": "Pre-trained weights checkpoint."},
    )


    def __post_init__(self):
        '''
        Initialize the training arguments and set up the log directory and job ID.
        If the job ID is set to "auto", it will automatically determine the next 
            available job ID based on existing directories.
        The log directory will be created based on the job ID and the current timestamp.
        If tee_output is True, it will redirect stdout and stderr to a log file 
            while also displaying the output on the console.
        '''
        if self.job_id == "auto":
            assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 1, "Might be problematic with DDP."
            if Path(self.log_dir).exists() and len(os.listdir(self.log_dir)) > 0:        
                next_job_id = str(max([int(x.name) for x in Path(self.log_dir).iterdir() if x.name.isnumeric()])+1)
            else:
                next_job_id = "1"
            self.job_id = next_job_id

        self.log_dir = os.path.join(
            self.log_dir, self.job_id, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        os.makedirs(self.log_dir, exist_ok=True)
             
