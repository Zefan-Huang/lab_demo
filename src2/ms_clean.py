import pandas as pd

def load_data():
    df = pd.read_csv('output/analysis_result.csv')
    return df

def drop_columns(df):
    df = df.loc[:, ['fitmri_id', 'total_steps']]
    print(df.head())
    return df

def add_label(df):
    df['label'] = 0
    df = df.rename(columns={'fitmri_id': 'id'})
    df = df.rename(columns={'total_steps': 'steps'})
    print(df.head())
    return df

def save(df):
    df.to_csv('unsupervised/cleaned_MS_data.csv', index=False)
    print("MS data Saved")

def main():
    df = load_data()
    df = drop_columns(df)
    df = add_label(df)
    save(df)

if __name__ == "__main__":
    main()
