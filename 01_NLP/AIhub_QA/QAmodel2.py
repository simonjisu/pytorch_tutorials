import torch
import pytorch_lightning as pl
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
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

# typing
from transformers.data.processors import SquadFeatures
from typing import List

def flatten(li):
    for ele in li:
        if isinstance(ele, list):
            yield from flatten(ele)
        else:
            yield ele

class ModelOriginal(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters() 
        # Create Tokenizer
        self.tokenizer = ElectraTokenizer.from_pretrained(self.hparams.model_name_or_path)
        # Create dataset and cache it
        self.train_files = []
        self.val_files = []
        self.create_dataset_all(state="train")
        self.create_dataset_all(state="val")
        self.all_examples, self.all_features = None, None
        # Create Model
        self.config = ElectraConfig.from_pretrained(self.hparams.model_name_or_path)
        self.model = ElectraForQuestionAnswering.from_pretrained(
            self.hparams.model_name_or_path, 
            config=self.config
        )

    def to_list(self, x):
        return x.detach().cpu().tolist()

    def create_dataset_all(self, state:str):
        cache_file = self.hparams.cache_file.format(state, "all")
        processed_file = self.hparams.ckpt_path / cache_file

        if processed_file.exists():
            print(f"[INFO] cache file already exists! passing the procedure")
            print(f"[INFO] Path: {processed_file}")
            if state == "train":
                self.train_files.append(cache_file)
            elif state == "val":
                self.val_files.append(cache_file)
            else:
                raise ValueError("state should be train or val")
            return None
        else:
            processor = SquadV2Processor()
            if state == "train":
                filename = self.hparams.train_file
                process_fn = processor.get_train_examples
                is_training = True
                self.train_files.append(cache_file)
            elif state == "val":
                filename = self.hparams.val_file
                process_fn = processor.get_dev_examples
                is_training = False
                self.val_files.append(cache_file)
            else:
                raise ValueError("state should be train or val")
            
            print(f"[INFO] Processing: {filename} | Cache file name: {cache_file}")
            examples = process_fn(
                data_dir=self.hparams.data_path, 
                filename=filename
            )
            features = squad_convert_examples_to_features(
                examples=examples,
                tokenizer=self.tokenizer,
                max_seq_length=self.hparams.max_seq_length,
                doc_stride=self.hparams.doc_stride,
                max_query_length=self.hparams.max_query_length,
                is_training=is_training,
                return_dataset=False,
                threads=self.hparams.threads,
            )
            dataset = self.convert_to_tensor(state, features)
            cache = dict(dataset=dataset, examples=examples, features=features)
            torch.save(cache, processed_file)
            print(f"[INFO] cache file saved! {processed_file}")

    def convert_to_tensor(self, state:str, features:List[SquadFeatures]):
        """
        Reference: https://github.com/huggingface/transformers/blob/master/src/transformers/data/processors/squad.py
        Arguments:
            state {str} -- [description]
        """
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        
        if state == "train":
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids, all_attention_masks, all_token_type_ids, all_start_positions, all_end_positions
            )
        elif state == "val":
            all_unique_ids = torch.tensor([f.unique_id for f in features], dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids, all_attention_masks, all_token_type_ids, all_unique_ids
            )
        else:
            raise ValueError("state should be train or val")
        return dataset

    def load_cache(self, filename:str, return_dataset:bool=True):
        processed_file = self.hparams.ckpt_path / filename
        cache = torch.load(processed_file)
        dataset, examples, features = cache["dataset"], cache["examples"], cache["features"]

        if return_dataset:
            return dataset
        else:
            return examples, features

    def create_dataloader(self, state:str="train"):
        if state == "train":
            shuffle = True
            batch_size = self.hparams.train_batch_size
            filename = self.train_files[0]
        elif state == "val":
            shuffle = False
            batch_size = self.hparams.eval_batch_size
            filename = self.val_files[0]
        else:
            raise ValueError("state should be train or val")

        dataset = self.load_cache(filename=filename, return_dataset=True)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.threads
        )
        return dataloader

    def train_dataloader(self):
        return self.create_dataloader(state="train")

    def val_dataloader(self):
        return self.create_dataloader(state="val")

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def training_step(self, batch, batch_idx):
        inputs_ids, attention_mask, token_type_ids, start_positions, end_positions = batch

        outputs = self(
            input_ids=inputs_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            start_positions=start_positions,
            end_positions=end_positions
        )

        loss = outputs.loss
        return  {'loss': loss}

    def validation_step(self, batch, batch_idx, dataloader_idx):
        # batch = single dataloader batch not multiple dataloader
        inputs_ids, attention_mask, token_type_ids, data_unique_ids = batch
        outputs = self(
            input_ids=inputs_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            start_positions=None,
            end_positions=None
        )

        # outputs.values: [(B, H), (B, H)] > batch_results: (B, 2, H)
        # B = len(datasets) * batch_size
        batch_results = []
        for i, unique_id in enumerate(self.to_list(data_unique_ids)):
            output = [self.to_list(o[i]) for o in outputs.values()]
            start_logits, end_logits = output
            result = SquadResult(unique_id, start_logits, end_logits)
            batch_results.append(result)

        # for i, example_index in enumerate(example_indices):
        #     eval_feature = self.eval_features[example_index.item()]
        #     unique_id = int(eval_feature.unique_id)
        #     output = [self.tolist(o[i]) for o in outputs.values()]
        #     start_logits, end_logits = output
        #     result = SquadResult(unique_id, start_logits, end_logits)
        #     batch_results.append(result)
            
        return batch_results
    
    def train_epoch_end(self, outputs):
        loss = torch.tensor(0, dtype=torch.float)
        for out in outputs:
            loss += out["loss"].detach().cpu()
        loss = loss / len(outputs)

        return {'loss': loss}

    def validation_epoch_end(self, outputs):
        if (self.all_examples is None) or (self.all_features is None):
            examples, features = self.load_cache(filename=self.val_files[0], return_dataset=False)
            self.all_examples= examples
            self.all_features= features
            del examples
            del features

        all_results = list(flatten(outputs))
        # https://huggingface.co/transformers/_modules/transformers/data/processors/squad.html
        # TODO: Cannot find the key unique_id
        # BUG: must set argument of `trainer: num_sanity_val_steps=0` to avoid error.

        predictions = compute_predictions_logits(
            self.all_examples,
            self.all_features,
            all_results,
            self.hparams.n_best_size,
            self.hparams.max_answer_length,
            self.hparams.do_lower_case,
            self.hparams.ckpt_path / self.hparams.output_prediction_file.format(self.global_step),
            self.hparams.ckpt_path / self.hparams.output_nbest_file.format(self.global_step),
            self.hparams.ckpt_path / self.hparams.output_null_log_odds_file.format(self.global_step),
            self.hparams.verbose_logging,
            self.hparams.version_2_with_negative,
            self.hparams.null_score_diff_threshold,
            self.tokenizer,
        )
        results = squad_evaluate(self.all_examples, predictions)
        accuracy = results["exact"]
        f1 = results["f1"]
        self.log("val_acc", accuracy, on_epoch=True, prog_bar=True)
        self.log("val_f1", f1, on_epoch=True, prog_bar=True)
        # return {"val_acc": accuracy, "val_f1": f1}

    def configure_optimizers(self):
        t_total = self.total_steps()
        
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 
                "weight_decay": 0.0
            },
        ]
        optimizer = AdamW(
            params=optimizer_grouped_parameters, 
            lr=self.hparams.learning_rate, 
            eps=self.hparams.adam_epsilon
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer, 
            num_warmup_steps=int(t_total * self.hparams.warmup_proportion), 
            num_training_steps=t_total
        )
        
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }

    def total_steps(self):
        r"""
        source: https://github.com/PyTorchLightning/pytorch-lightning/issues/1038
        """
        return len(self.train_dataloader()) * self.hparams.num_train_epochs