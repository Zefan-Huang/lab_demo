import pandas as pd
import numpy as np

## print(pd.__version__)

class dataCheck:
    def __init__(self):
        self.df = pd.read_csv('data/FitMRI_fitbit_intraday_steps_trainingData.csv')
        print("Datset has been loaded")

    def take_look_data(self):
        how_many_patient = self.df['fitmri_id'].unique()
        total_num = len(how_many_patient)
        print("The list of patient:", how_many_patient)
        print("The total number of patient:", total_num)
        print("Type of id:", self.df['fitmri_id'].dtype)

    def clean_datetime(self):
        self.df['datetime'] = pd.to_datetime(self.df['measured_date'] + ' ' + self.df['measured_time'], format = '%d-%b-%y %H:%M:%S')
        self.df['measured_date'] = self.df['datetime'].dt.date
        print(self.df.head(10))
    
    def data_summary(self):
        min_max = self.df.groupby('fitmri_id')['datetime'].agg(['min', 'max']).reset_index()
        print(min_max)
    
    def filling_missing(self):
        missing = []
        for i in range(len(self.df)-1):
            cur = self.df['fitmri_id'].iloc[i]
            nxt = self.df['fitmri_id'].iloc[i+1]
            if nxt - cur > 1:
                gap = list(range(cur+1, nxt))
                missing.extend(gap)
        print("Missing IDs:", missing)
        return missing
    
    def save_cleaned_data(self, filename='data/cleaned_data.csv'):
        self.df.to_csv(filename, index=False)
        print(f"Cleaned data saved to {filename}")

    def run_all(self):
        check.take_look_data()
        check.clean_datetime()
        check.data_summary()
        check.filling_missing()
        check.save_cleaned_data()
        print(check.df.head())



check = dataCheck()
check.run_all()

