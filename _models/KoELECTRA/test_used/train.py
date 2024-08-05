from _utils.koelectra.tokenizer import get_tokenizer
from _utils.common.config import Config
from _embeddingModels.koelectra.network import KoElectra
from _embeddingModels.koelectra.data_processing import generate_training_data
from _embeddingModels.koelectra.trainer_noCRF import Trainer
from _utils.koelectra.utils import init_logger, set_seed


if __name__ == "__main__":
    # *** Train ***
    set_seed()
    init_logger()
    config = Config(json_path="./config.json")

    # tokenizer load
    tokenizer = get_tokenizer()
    # mode = "test" --> test data 사용할 때
    # mode = None --> train data split하여 사용할 때
    tr_dataloader, val_dataloader, label_list, vocab = generate_training_data(config=config, tokenizer=tokenizer, mode="test")
    model = KoElectra(config, num_labels=len(vocab))
    trainer = Trainer(config, tr_dataloader, val_dataloader, model, label_list=label_list)

    trainer.train()
