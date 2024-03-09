from imports import *

def split_data(df_train_val):
    train_val_files_lst = []
    images_lst = sorted(df_train_val['Images'].tolist())
    train_val_files_lst = [Path(item) for sublist in images_lst for item in sublist]
    
    train_val_labels = [path.parent.name for path in train_val_files_lst]
    train_files, val_files = train_test_split(train_val_files_lst, test_size=0.2, random_state=random_seed, stratify=train_val_labels)
    
    return train_files, val_files