import os
import pandas as pd

path_dir = "label/UCF-101"

def new_csv(path_dir):
    '''
    [Args]
    path_dir : path to folder contain train.csv, val.csv and newClassInd.txt which hold labels of class remained

    [Result]
    Create new_train.csv and new_val.csv save at path_dir 

    '''
    # Read path
    train = os.path.join(path_dir,'train.csv')
    val = os.path.join(path_dir,'val.csv')
    new_class = os.path.join(path_dir,'newClassInd.txt')
    new_train = os.path.join(path_dir,'new_train.csv')
    new_val = os.path.join(path_dir,'new_val.csv')

    if not os.path.exists(new_train):
        open(new_train, 'w').close() 
    if not os.path.exists(new_val):
        open(new_val, 'w').close() 

    df_csv_train= pd.read_csv(train)
    df_csv_val = pd.read_csv(val)

    # Take lables to keep in new file csv
    with open(new_class,'r') as f:
        labels_keep = set(line.strip().split(' ')[1] for line in f)

    # Filter train.csv
    df_filtered_train = df_csv_train[df_csv_train['label'].isin(labels_keep)]
    df_filtered_train.to_csv(new_train, index=False)

    # Filter val.csv
    df_filtered_val = df_csv_val[df_csv_val['label'].isin(labels_keep)]
    df_filtered_val.to_csv(new_val, index=False)

new_csv(path_dir)


