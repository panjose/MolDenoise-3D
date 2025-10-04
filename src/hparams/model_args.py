from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    '''Model arguments for the training script.'''

    # model architecture
    model: str = field(
        default="equivariant-transformer",
        metadata={"help": "Which model to train."},
    )

    # architectural args
    embedding_dimension: int = field(
        default=256,
        metadata={"help": "Embedding dimension."},
    )
    num_layers: int = field(
        default=6,
        metadata={"help": "Number of interaction layers in the model."},
    )
    num_rbf: int = field(
        default=64,
        metadata={"help": "Number of radial basis functions in model."},
    )
    num_heads: int = field(
        default=8,
        metadata={"help": "Number of attention heads."},
    )

    # other args
    derivative: bool = field(
        default=False,
        metadata={"help": "If true, take the derivative of the prediction w.r.t coordinates."},
    )
    cutoff_lower: float = field(
        default=0.0,
        metadata={"help": "Lower cutoff in model."},
    )
    cutoff_upper: float = field(
        default=5.0,
        metadata={"help": "Upper cutoff in model."},
    )
    atom_filter: int = field(
        default=-1,
        metadata={"help": "Only sum over atoms with Z > atom_filter."},
    )
    max_z: int = field(
        default=100,
        metadata={"help": "Maximum atomic number that fits in the embedding matrix."},
    )
    max_num_neighbors: int = field(
        default=32,
        metadata={"help": "Maximum number of neighbors to consider in the network."},
    )
    standardize: bool = field(
        default=False,
        metadata={"help": "If true, multiply prediction by dataset std and add mean."},
    )


    def __post_init__(self):
        pass
