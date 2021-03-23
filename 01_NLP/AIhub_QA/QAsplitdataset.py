import json
from tqdm import tqdm
from QAtrainer import create_args

def split_dataset(args_dict:dict, n_split:int=10):
    data_path = args_dict["data_path"]
    
    processd_length = []
    for path in data_path.glob("*all.json"):
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

if __name__ == "__main__":
    args_dict = create_args()
    split_dataset(args_dict, n_split=10)