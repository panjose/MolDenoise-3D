from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class DataArguments:
    '''Data arguments for the training script.'''

    dataset: Literal["QM9", "MD17", "ANI1", "Custom", "HDF5", "PCQM4MV2"] = field(
        default=None,
        metadata={"help": "Name of the torch_geometric dataset."},
    )
    dataset_root: Optional[str] = field(  
        default='data',
        metadata={"help": "Data storage directory (not used if dataset is 'CG')."},
    )
    dataset_arg: Optional[str] = field(
        default=None,
        metadata={"help": "Additional dataset argument, e.g. target property for QM9 or molecule for MD17."},
    )
    energy_weight: float = field(
        default=1.0,
        metadata={"help": "Weighting factor for energies in the loss function."},
    )
    force_weight: float = field(
        default=1.0,
        metadata={"help": "Weighting factor for forces in the loss function."},
    )
    position_noise_scale: float = field(
        default=0.0,
        metadata={"help": "Scale of Gaussian noise added to positions."},
    )
    denoising_weight: float = field(
        default=0.0,
        metadata={"help": "Weighting factor for denoising in the loss function."},
    )
    denoising_only: bool = field(
        default=False,
        metadata={"help": "If the task is denoising only (then val/test datasets also contain noise)."},
    )


    def __post_init__(self):
        pass
