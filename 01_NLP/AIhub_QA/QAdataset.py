import json
import numpy as np
from tqdm import tqdm
from QAtrainer import create_args
from typing import Dict, List, Any

def test(args_dict:Dict[str, Any]):
    data_path = args_dict["data_path"]
    for path in data_path.glob("ko_nia*all_preprocessed.json"):
        with open(path, 'rb') as f:
            squad_dict = json.load(f)
        for i, group in tqdm(enumerate(squad_dict["data"]), total=len(squad_dict["data"]), desc="Reading datset"):
            for paragraph in group["paragraphs"]:
                context = paragraph["context"]
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    if qas_id in ["m6_415182-2", "m6_413880-2", "m6_413136-2"]:
                        print(path.name, qas_id)
                        

def preprocess_dataset(args_dict:Dict[str, Any]):
    """delete all blank question and context"""
    data_path = args_dict["data_path"]
    for path in data_path.glob("ko_nia*all.json"):
        with open(path, 'rb') as f:
            squad_dict = json.load(f)
        missing_context_ids = []
        missing_qas_ids = []
        for i, group in tqdm(enumerate(squad_dict["data"]), total=len(squad_dict["data"]), desc="Reading datset"):
            for paragraph in group["paragraphs"]:
                context = paragraph["context"]
                if context.strip() == "":
                    missing_context_ids.append(i)
                    continue
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    if question_text.strip() == "":
                        missing_qas_ids.append(qas_id)
        for i in range(len(squad_dict["data"])):
            if i in missing_context_ids:
                squad_dict["data"].pop(i)
                print(f"[{path.name}] data {i}")
                continue
            qas = squad_dict["data"][i]["paragraphs"][0]["qas"]
            for j, qa in enumerate(qas):
                if qa["id"] in missing_qas_ids:
                    squad_dict["data"][i]["paragraphs"][0]["qas"].pop(j)
                    print(f"[{path.name}] dropped id: {qa['id']}")
        p = data_path / (path.name.strip(".json") + "_preprocessed.json")
        with open(p, "w") as f:
            json.dump(squad_dict, f)

def split_dataset(args_dict:Dict[str, Any], n_split:int=10):
    data_path = args_dict["data_path"]
    
    processd_length = []
    for path in data_path.glob("ko_nia*all.json"):
        if path.name == "ko_nia_normal_squad_all.json":
            state = "train"
        elif path.name == "ko_nia_clue0529_squad_all.json":
            state = "val"
        else:
            continue
        # read
        with open(path, 'rb') as f:
            squad_dict = json.load(f)
        total_examples = len(squad_dict["data"])
        k = len(squad_dict["data"]) // n_split
        processed = 0
        print(f"[INFO] Start to split: {path.name}")
        for i in tqdm(range(n_split), total=n_split, desc=f"Spliting {state}"):
            p = data_path / f"{state}_{i}.json"
            temp_data = dict(
                creator=squad_dict["creator"], 
                version=squad_dict["version"], 
                data=squad_dict["data"][i:i+k]
            )
            processed += k
            with open(p, "w") as f:
                json.dump(temp_data, f)
                
        if processed < total_examples:
            p = data_path / f"{state}_{i+1}.json"
            temp_data = dict(
                creator=squad_dict["creator"], 
                version=squad_dict["version"], 
                data=squad_dict["data"][processed:]
            )
            with open(p, "w") as f:
                json.dump(temp_data, f)

def filter_dataset(args_dict:Dict[str, Any]):
    data_path = args_dict["data_path"]
    model_name_or_path = args_dict["model_name_or_path"]
    from transformers import ElectraTokenizer
    tokenizer = ElectraTokenizer.from_pretrained(model_name_or_path)
    
    length_data = {}
    for path in data_path.glob("ko_nia*all.json"):
        if path.name == "ko_nia_normal_squad_all.json":
            state = "train"
        elif path.name == "ko_nia_clue0529_squad_all.json":
            state = "val"
        else:
            continue
        with open(path, 'rb') as f:
            squad_dict = json.load(f)
        length_of_texts = []
        for group in tqdm(squad_dict["data"], total=len(squad_dict["data"]), desc="Getting Length of Tokenized Context"):
            for paragraph in group["paragraphs"]:
                context = paragraph["context"]
                length_of_texts.append(len(tokenizer.encode(context)))
        length_of_texts = np.array(length_of_texts)
        for token_length in [512, 1024]:
            print(f"Processing: {state}/{token_length}")
            filter_n = token_length if token_length == 512 else token_length*2
            idx_over = np.arange(len(length_of_texts))[length_of_texts >= filter_n]
            print(f"count: # of over {token_length} is {len(idx_over)}/{len(length_of_texts)}, percentage: {len(idx_over)/len(length_of_texts)*100:.2f}%")

            choose_data = []
            for i in tqdm(range(len(squad_dict["data"])), total=len(squad_dict["data"]), desc=f"{state}, {token_length}"):
                if i in idx_over:
                    continue
                else:
                    choose_data.append(squad_dict["data"][i])
            temp_data = dict(
                creator=squad_dict["creator"], 
                version=squad_dict["version"], 
                data=choose_data
            )
            p = data_path / f"AIhub_squad_{state}_{token_length}.json"
            with open(p, "w") as f:
                json.dump(temp_data, f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="process dataset")
    parser.add_argument("--type", type=str,
        help="type to process either filter or split or preprocess")
    args = parser.parse_args()
    args_dict = create_args()
    
    if args.type == "filter":
        filter_dataset(args_dict)
    elif args.type == "split":
        split_dataset(args_dict, n_split=10)
    elif args.type == "preprocess":
        preprocess_dataset(args_dict)
    elif args.type == "test":
        test(args_dict)
    else:
        raise ValueError("must set --type argument")