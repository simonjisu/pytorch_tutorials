from pathlib import Path
import torch
import pytorch_lightning as pl

from QAmodel import Model
from QAmodel2 import ModelOriginal

# RuntimeError: Too many open files. Communication with the workers is no longer possible. 
# Please increase the limit using `ulimit -n` in the shell or change the sharing strategy by calling 
# `torch.multiprocessing.set_sharing_strategy('file_system')` at the beginning of your code
torch.multiprocessing.set_sharing_strategy("file_system")

def create_args():
    models = {
        "splitted": Model,
        "original": ModelOriginal
    }
    model_type = "original"
    train_file = "train*.json" if model_type == "splitted" else "ko_nia_normal_squad_all_preprocessed.json" # "AIhub_squad_train_512.json"
    val_file = "val*.json" if model_type == "splitted" else "ko_nia_clue0529_squad_all_preprocessed.json" # "AIhub_squad_val_512.json"

    repo_path = Path().absolute()
    data_path = repo_path.parent / "data" / "AIhub" / "QA"
    ckpt_path = repo_path.parent / "ckpt"
    
    args_dict = {
        "model_class": models[model_type],
        "task": "AIhub_QA",
        "data_path": data_path,
        "ckpt_path": ckpt_path,
        "train_file": train_file,
        "val_file": val_file,
        "cache_file": "{}_cache_{}",
        "random_seed": 77,
        "threads": 16,
        "version_2_with_negative": False,
        "null_score_diff_threshold": 0.0,
        "max_seq_length": 512,
        "doc_stride": 128,
        "max_query_length": 64,
        "max_answer_length": 30,
        "n_best_size": 20,
        "verbose_logging": True,
        "do_lower_case": False,
        "num_train_epochs": 7,
        "weight_decay": 0.0,
        "adam_epsilon": 1e-8,
        "warmup_proportion": 0,
        "model_type": "koelectra-base-v3",
        "model_name_or_path": "monologg/koelectra-base-v3-discriminator",
        "output_dir": "koelectra-base-v3-korquad-ckpt",
        "train_batch_size": 1 if model_type == "splitted" else 8,
        "eval_batch_size": 1 if model_type == "splitted" else 8,
        "learning_rate": 5e-5,
        "output_prediction_file": "predictions/predictions_{}.json",
        "output_nbest_file": "nbest_predictions/nbest_predictions_{}.json",
        "output_null_log_odds_file": "null_odds/null_odds_{}.json",
    }

    # Path Check
    # if not ckpt_path.exists():
    #     ckpt_path.mkdir()
    # else:
    #     if rebuild:
    #         for x in ckpt_path.glob("*"):
    #             if x.is_dir():
    #                 x.rmdir()
    #             else:
    #                 x.unlink()
    #         ckpt_path.rmdir()
    #         ckpt_path.mkdir()

    for arg in ["output_prediction_file", "output_nbest_file", "output_null_log_odds_file", ]:
        p = args_dict["ckpt_path"] / args_dict[arg]
        if not p.parent.exists():
            p.parent.mkdir()
    # for tensorboard logger
    if not (args_dict["ckpt_path"] / args_dict["task"]).exists():
        (args_dict["ckpt_path"] / args_dict["task"]).mkdir()
    return args_dict

def main(args_dict):
    print("[INFO] Using PyTorch Ver", torch.__version__)
    print("[INFO] Seed:", args_dict["random_seed"])
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="epoch{epoch:02d}-f1{f1:.4f}",
        # dirpath=args_dict["ckpt_path"],
        monitor="val_f1",
        save_top_k=3,
        mode="max",
    )
    earlystop_callback = pl.callbacks.EarlyStopping("val_f1", mode="max")
    pl.seed_everything(args_dict["random_seed"])
    model = args_dict["model_class"](**args_dict)
    logger = pl.loggers.TensorBoardLogger(str(args_dict["ckpt_path"]), name=args_dict["task"])
    
    print("[INFO] Start FineTuning")
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, earlystop_callback],
        default_root_dir=args_dict["ckpt_path"],
        max_epochs=args_dict["num_train_epochs"],
        deterministic=torch.cuda.is_available(),
        gpus=-1 if torch.cuda.is_available() else None,
        accelerator="ddp",
        num_sanity_val_steps=0,
        logger=logger
    )
    trainer.fit(model)

if __name__ == "__main__":
    args_dict = create_args()
    main(args_dict)