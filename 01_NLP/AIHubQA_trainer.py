from pathlib import Path
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchmetrics

from transformers import (
    ElectraForQuestionAnswering, 
    ElectraConfig, 
    ElectraTokenizer,
    AdamW,
    squad_convert_examples_to_features,
    get_linear_schedule_with_warmup
)

from transformers.data.processors.squad import SquadResult, SquadV2Processor
from transformers.data.metrics.squad_metrics import (
    compute_predictions_logits,
    squad_evaluate
)
from .AIHubQA_model import Model

def create_args():
    train_file = "ko_nia_normal_squad_all.json"
    val_file = "ko_nia_clue0529_squad_all.json"

    repo_path = Path().absolute().parent
    data_path = repo_path.parent / "data" / "AIhub" / "QA"
    ckpt_path = repo_path.parent / "ckpt"
    if not ckpt_path.exists():
        ckpt_path.mkdir()
    else:
        for x in ckpt_path.glob("*"):
            if x.is_dir():
                x.rmdir()
            else:
                x.unlink()
        ckpt_path.rmdir()
        ckpt_path.mkdir()

    args_dict = {
        "task": "AIHub_QA",
        "data_path": data_path,
        "ckpt_path": ckpt_path,
        "train_file": train_file,
        "val_file": val_file,
        "cache_file": "{}_cache",
        "random_seed": 77,
        "threads": 4,
        "version_2_with_negative": False,
        "null_score_diff_threshold": 0.0,
        "max_seq_length": 512,
        "doc_stride": 128,
        "max_query_length": 64,
        "max_answer_length": 30,
        "n_best_size": 20,
        "verbose_logging": True,
        "do_lower_case": False,
        "num_train_epochs": 10,
        "weight_decay": 0.0,
        "adam_epsilon": 1e-8,
        "warmup_proportion": 0,
        "model_type": "koelectra-base-v3",
        "model_name_or_path": "monologg/koelectra-base-v3-discriminator",
        "output_dir": "koelectra-base-v3-korquad-ckpt",
        "seed": 42,
        "train_batch_size": 8,
        "eval_batch_size": 8,
        "learning_rate": 5e-5,
        "output_prediction_file": "predictions_{}.json",
        "output_nbest_file": "nbest_predictions_{}.json",
        "output_null_log_odds_file": "null_odds_{}.json",
    }

return args_dict

def main(args_dict):
    print("[INFO] Using PyTorch Ver", torch.__version__)
    print("[INFO] Seed:", args_dict["random_seed"])
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="epoch{epoch}-f1{f1:.4f}",
        monitor="f1",
        save_top_k=3,
        mode="max",
    )
    pl.seed_everything(args_dict["random_seed"])
    model = Model(**args_dict)
    
    print("[INFO] Start FineTuning")
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=args_dict["num_train_epochs"],
        deterministic=torch.cuda.is_available(),
        gpus=-1 if torch.cuda.is_available() else None,
    )
    trainer.fit(model)

if __name__ == "__main__":
    args_dict = create_args()
    main(args_dict)