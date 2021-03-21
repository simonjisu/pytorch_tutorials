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
        self.create_dataset(state="train")
        self.create_dataset(state="val")
        self.eval_examples, self.eval_features = self.load_cache(state="val", return_dataset=False)
        # function
        self.tolist = lambda x: x.detach().cpu().tolist()
        
    def create_dataset(self, state:str="train"):
        r"""
        Args:
            state: train or val
        """
        processor = SquadV2Processor()
        if state == "train":
            examples = processor.get_train_examples(
                data_dir=self.hparams.data_path, 
                filename=self.hparams.train_file
            )
            is_training = True
        elif state == "val":
            examples = processor.get_dev_examples(
                data_dir=self.hparams.data_path, 
                filename=self.hparams.val_file
            )
            is_training = False
        else:
            raise ValueError("state should be train or val")
            
        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=self.tokenizer,
            max_seq_length=self.hparams.max_seq_length,
            doc_stride=self.hparams.doc_stride,
            max_query_length=self.hparams.max_query_length,
            is_training=is_training,
            return_dataset="pt",
            threads=self.hparams.threads,
        )
        # https://huggingface.co/transformers/_modules/transformers/data/processors/squad.html
        # TODO: Cannot find the key unique_id
        # BUG: must set argument of `trainer: num_sanity_val_steps=0` to avoid error.

        cache = dict(dataset=dataset, examples=examples, features=features)
        torch.save(cache, self.hparams.ckpt_path / self.hparams.cache_file.format(state))
        print(f"[INFO] cache file saved! {self.hparams.ckpt_path / self.hparams.cache_file.format(state)}")

    def load_cache(self, state:str="train", return_dataset:bool=True):
        cache = torch.load(self.hparams.ckpt_path / self.hparams.cache_file.format(state))
        dataset, examples, features = cache["dataset"], cache["examples"], cache["features"]

        if return_dataset:
            return dataset
        else:
            return examples, features

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def training_step(self, batch, batch_idx):
        inputs_ids, attention_mask, token_type_ids, start_positions, end_positions, *_ = batch

        outputs = self(
            input_ids=inputs_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            start_positions=start_positions,
            end_positions=end_positions
        )

        loss = outputs.loss
        return  {'loss': loss}

    def validation_step(self, batch, batch_idx):
        inputs_ids, attention_mask, token_type_ids, example_indices, *_ = batch
        
        outputs = self(
            input_ids=inputs_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            start_positions=None,
            end_positions=None
        )
        
        batch_results = []
        
        for i, example_index in enumerate(example_indices):
            eval_feature = self.eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            output = [self.tolist(o[i]) for o in outputs.values()]
            start_logits, end_logits = output
            result = SquadResult(unique_id, start_logits, end_logits)
            batch_results.append(result)
            
        return batch_results
    
    def train_epoch_end(self, outputs):
        loss = torch.tensor(0, dtype=torch.float)
        for out in outputs:
            loss += out["loss"].detach().cpu()
        loss = loss / len(outputs)

        return {'loss': loss}

    def validation_epoch_end(self, outputs):
        all_results = []
        for res in outputs:
            all_results += res

        predictions = compute_predictions_logits(
            self.eval_examples,
            self.eval_features,
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
        results = squad_evaluate(self.eval_examples, predictions)
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

    def create_dataloader(self, state:str="train"):
        r"""
        Args:
            state: train or val
        """
        if state == "train":
            shuffle = True
            batch_size = self.hparams.train_batch_size
        elif state == "val":
            shuffle = False
            batch_size = self.hparams.eval_batch_size
        else:
            raise ValueError("state should be train or val")
        dataset = self.load_cache(state, return_dataset=True)
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
    
    def total_steps(self):
        r"""
        source: https://github.com/PyTorchLightning/pytorch-lightning/issues/1038
        """
        return len(self.train_dataloader()) * self.hparams.num_train_epochs