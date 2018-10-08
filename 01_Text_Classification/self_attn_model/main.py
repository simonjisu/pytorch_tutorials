import config
import pandas as pd
from toxic_train import preprocessing, import_data, build_model, train_model

def main():
    train_data = pd.read_csv(config.PATH + config.TRAIN_PATH)
    preprocessing(train_data, save_path=config.PATH + 'train_data.tsv', split_rt=config.SPLIT, valid_path=config.PATH + 'valid_data.tsv')
    train, valid, COMMENT, LABEL, train_iter, valid_iter = import_data(config)
    model, loss_function, optimizer, scheduler = build_model(config, comment_field=COMMENT)
    train_model(config, model, train_iter, valid_iter, loss_function, optimizer, scheduler)
    
if __name__ == '__main__':
    main()