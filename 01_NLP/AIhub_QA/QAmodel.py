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

class Model(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters() 
        self.config = ElectraConfig.from_pretrained(self.hparams.model_name_or_path)
        self.model = ElectraForQuestionAnswering.from_pretrained(
            self.hparams.model_name_or_path, 
            config=self.config
        )
        self.tokenizer = ElectraTokenizer.from_pretrained(self.hparams.model_name_or_path)
        # create dataset and cache it
        self.train_files = []
        self.val_files = []
        self.create_dataset_all(state="train")
        self.create_dataset_all(state="val")

        self.all_examples, self.all_features = [], []

    def to_list(self, x):
        return x.detach().cpu().tolist()

    def create_dataset_all(self, state:str):
        self.example_index = 0
        self.unique_id = 1000000000
        if state == "train":
            file_str = self.hparams.train_file
        elif state == "val":
            file_str = self.hparams.val_file
        else:
            raise ValueError("state should be train or val")
        
        file_iter = sorted(self.hparams.data_path.glob(file_str), key=lambda x: int(x.name.strip(".json").split("_")[-1]))
        for path in file_iter:
            filename = path.name
            idx = int(filename.strip(".json").split("_")[-1])
            self.create_dataset(path.name, idx, state)

    def create_dataset(self, filename:str, idx:int, state:str):
        cache_file = self.hparams.cache_file.format(state, idx)
        print(f"[INFO] Processing: {filename} | Cache file name: {cache_file}")
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
                process_fn = processor.get_train_examples
                is_training = True
                self.train_files.append(cache_file)
            elif state == "val":
                process_fn = processor.get_dev_examples
                is_training = False
                self.val_files.append(cache_file)
            else:
                raise ValueError("state should be train or val")

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
            # need to fix all `example_index` and `unique_id` since splitted the dataset only on validation dataset
            self.fix_unique_id(features, state)
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

    def fix_unique_id(self, features:list, state:str="val"):
        if state == "val":
            previous_example_index = -1 
            for fea in tqdm(features, total=len(features), desc="fixing index and ids"):
                fea.unique_id = self.unique_id
                self.unique_id += 1
                
                current_example_index = fea.example_index
                if previous_example_index == current_example_index:
                    fea.example_index = previous_example_index
                else:
                    previous_example_index = fea.example_index
                    fea.example_index = self.example_index
                    self.example_index += 1
        else:
            return None

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
            file_list = self.train_files
        elif state == "val":
            shuffle = False
            batch_size = self.hparams.eval_batch_size
            file_list = self.val_files
        else:
            raise ValueError("state should be train or val")

        caches = [self.load_cache(filename=file, return_dataset=True) for file in file_list]
        loaders = []
        for dataset in caches: 
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=self.hparams.threads
            )
            loaders.append(dataloader)
        return loaders

    def train_dataloader(self):
        return self.create_dataloader(state="train")

    def val_dataloader(self):
        return self.create_dataloader(state="val")

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def training_step(self, batch, batch_idx):
        batch = list(map(torch.cat, zip(*batch)))
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
        if (self.all_examples == []) or (self.all_features == []):
            for file in self.val_files:
                examples, features = self.load_cache(filename=file, return_dataset=False)
                self.all_examples.extend(examples)
                self.all_features.extend(features)
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
        self.log("accuracy", accuracy, on_epoch=True, prog_bar=True)
        self.log("f1", f1, on_epoch=True, prog_bar=True)

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