from src.hparams import get_args
from src.trainer import FradPretrainer


def run_pretrain():
    data_args, model_args, training_args = get_args()
    print("Data Arguments:", data_args)
    print("Model Arguments:", model_args)
    print("Training Arguments:", training_args)
    
    # Initialize the pretrainer
    trainer = FradPretrainer(data_args, model_args, training_args)
    trainer.train()
    print("Pretraining completed successfully.")

    
if __name__ == "__main__":
    run_pretrain()
