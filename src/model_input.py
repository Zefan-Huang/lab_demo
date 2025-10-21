import numpy as np
import pandas as pd

def load_data():
    df = pd.read_csv('output/analysis_result.csv')
    return df

def split_data(df, train_ratio):
    train_df = []
    test_df = []
    for pid, col in df.groupby('fitmri_id'):
        group = col.sort_values('measured_date')
        group = group[['fitmri_id', 'total_steps_normalized']]
        idx = int(len(group) * train_ratio)
        train_df.append(group[:idx])
        test_df.append(group[idx:])

    train_df = pd.concat(train_df, ignore_index=True)
    test_df = pd.concat(test_df, ignore_index=True)
    return train_df, test_df

def save_data(train_df, test_df):
    train_df.to_csv('output/train.csv', index=False)
    test_df.to_csv('output/test.csv', index=False)
    print('Train and Test data saved')

def main():
    df = load_data()
    train_df, test_df = split_data(df, 0.8)   # <------ if anyone wants to change the dataset ratio.
    save_data(train_df, test_df)

if __name__ == '__main__':
    main()