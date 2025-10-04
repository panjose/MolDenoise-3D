from src.hparams import get_args
from src.trainer import FradFinetuner


def run_finetune():
    data_args, model_args, training_args = get_args()
    print("Data Arguments:", data_args)
    print("Model Arguments:", model_args)
    print("Training Arguments:", training_args)
    
    # Initialize the finetuner
    trainer = FradFinetuner(data_args, model_args, training_args)
    trainer.train()
    print("Finetuning completed successfully.")

    
if __name__ == "__main__":
    run_finetune()
