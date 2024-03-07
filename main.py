from custom_dataset import SimpsonsDataset
from model_init import model_init
from read_data import read_data
from search_dub import search_dub
from split_data import split_data
from test import make_f1_score
from train import train

from imports import *


def main(): 
    df_train_val = read_data(train_val_dir) 

    df_duplicates = search_dub(train_val_dir)
    for index, row in df_duplicates.iterrows():
        df_train_val = df_train_val[~df_train_val['Images'].apply(lambda x: set(x) == set(row['Duplicate']))]

    train_files, val_files = split_data(df_train_val)
    train_dataset = SimpsonsDataset(train_files, mode="train")
    val_dataset = SimpsonsDataset(val_files, mode="val")

    dataloaders = {'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
                   'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False )}
    

    model = model_init(df_train_val)
    train(dataloaders,model)
    torch.save(model.state_dict(), output_model_dir+"model.pt")

    test_files = sorted(Path(test_dir).rglob('*.jpg'))
    test_dataset = SimpsonsDataset(test_files, mode="test")
    f1_score = make_f1_score(test_dataset)
    print(f"Test f1-score = {f1_score}.")


if __name__ == "__main__":
    main()