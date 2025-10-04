import os
import sys

from dataclasses import asdict
from transformers import HfArgumentParser
from typing import Tuple

from .data_args import DataArguments
from .model_args import ModelArguments
from .training_args import TrainingArguments
from ..utils import write_json


def _parse_args() -> Tuple[DataArguments, ModelArguments, TrainingArguments]:
    '''
    Parses command line arguments and returns instances of DataArguments, ModelArguments, and TrainingArguments.
    If a YAML file is provided as the first argument, it will parse the arguments from that file instead.
    If any unknown arguments are provided, it will raise a ValueError.
    Returns:
        Tuple[DataArguments, ModelArguments, TrainingArguments]: Parsed arguments as instances of the respective classes.
    Raises:
        ValueError: If there are unknown arguments that are not used by the HfArgumentParser.
    '''
    parser = HfArgumentParser([DataArguments, ModelArguments, TrainingArguments])

    if len(sys.argv) == 2 and sys.argv[1].endswith((".yaml", ".yml")):
        return parser.parse_yaml_file(sys.argv[1])

    (*parsed_args, unknown_args) = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )

    if unknown_args:
        print(parser.format_help())
        print(f"Got unknown args, potentially deprecated arguments: {unknown_args}")
        raise ValueError(
            f"Some specified arguments are not used by the HfArgumentParser: {unknown_args}"
        )

    return (*parsed_args,)


def get_args() -> Tuple[DataArguments, ModelArguments, TrainingArguments]:
    '''
    Parses command line arguments and returns instances of DataArguments, ModelArguments, and TrainingArguments.
    It also writes the parsed arguments to a JSON file in the log directory.
    Returns:
        Tuple[DataArguments, ModelArguments, TrainingArguments]: Parsed arguments as instances of the respective classes.
    '''
    data_args, model_args, training_args = _parse_args()
    config = {
        "data_args": asdict(data_args),
        "model_args": asdict(model_args),
        "training_args": asdict(training_args),
    }

    write_json(os.path.join(training_args.log_dir, "config.json"), config)
    return data_args, model_args, training_args

