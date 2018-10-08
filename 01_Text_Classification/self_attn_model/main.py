import config
from toxic_train import import_data, build_model, train_model

def main():
    train, test, COMMENT, LABEL, train_iter, test_iter = import_data(config)
    model, loss_function, optimizer, scheduler = build_model(config, comment_field=COMMENT)
    train_model(config, model, train_iter, test_iter, loss_function, optimizer, scheduler)
    
if __name__ == '__main__':
    main()