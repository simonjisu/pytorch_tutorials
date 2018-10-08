import sys
import config
import torch
import pandas as pd
from toxic_train import preprocessing, import_data, build_iteration, build_model
from torchtext.data import TabularDataset, Iterator

# config.DEVICE = None

train, valid, COMMENT, LABEL, _, _ = import_data(config)
model, loss_function, _, _ = build_model(config, comment_field=COMMENT)

test = pd.read_csv(config.PATH + 'test.csv').sort_values('id') 
test_labels = pd.read_csv(config.PATH + 'test_labels.csv').sort_values('id')
sub = pd.read_csv(config.PATH + 'sample_submission.csv')
test = pd.merge(test, test_labels, how='inner', on='id')
test.iloc[55142, 1] = '<unk>'
df_test = preprocessing(test, save_path=config.PATH + 'test_kaggle.tsv')


test_dataset = TabularDataset(config.PATH + 'test_kaggle.tsv', fields=[('cmt', COMMENT), ('lbl', LABEL)], format='tsv')
test_iter = Iterator(test_dataset, batch_size=1, device=config.DEVICE, shuffle=False, repeat=False)
model.load_state_dict(torch.load(config.SAVE_PATH))
all_preds = []
for i, batch in enumerate(test_iter):
    inputs, lengths = batch.cmt
    scores, _ = model(inputs, lengths, device=config.DEVICE)
    preds = torch.sigmoid(scores).view(-1).detach().tolist()
    all_preds.append(preds)
    sys.stdout.write('\r')
    sys.stdout.write('Predicting: {}, {:.2f}'.format(i, i/len(test_iter)*100))
    sys.stdout.flush()
result = pd.DataFrame(all_preds, columns=test_labels.columns[1:])
sub.iloc[:, 1:] = result
sub.to_csv(config.PATH + 'sub_kaggle.csv', index=False, header=True, sep=',')

    
    

