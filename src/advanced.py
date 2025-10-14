import pandas as pd
import numpy as np
from pandas import merge

# import basic_analysis as ba

def load_csv():
    df = pd.read_csv('data/cleaned_data.csv')
    return df

def summarize_data(df):
    summary = df.describe()
    na = df.isna().sum()
    print("Data Summary:\n", summary)
    print("Missing Values:\n", na)

def check_missing_hours(df):
    df['hour'] =pd.to_datetime(df['measured_time'], format='%H:%M:%S', errors='coerce').dt.hour
    hours = df.groupby(['fitmri_id', 'measured_date'])['hour'].nunique()
    hours = hours.reset_index()
    hours.rename(columns = {'hour': 'recorded_hours'}, inplace=True)
    return hours

def sum_steps_per_patient(df):
    sum_steps = df.groupby(['fitmri_id', 'measured_date'])['steps'].sum().reset_index()
    sum_steps.rename(columns={'steps': 'total_steps'}, inplace=True)
    return sum_steps

def hourly_average_steps(df):
    hours = check_missing_hours(df)
    steps = sum_steps_per_patient(df)
    merged = pd.merge(steps, hours, on=['fitmri_id', 'measured_date'], how='outer')
    merged['recorded_hours'] = merged['recorded_hours'].replace({0: np.nan})
    merged['hourly_avg_steps'] = (merged['total_steps'] / merged['recorded_hours']).round().astype('Int64')
    return merged

def is_weekend(df):
    df['measured_date'] = pd.to_datetime(df['measured_date'], errors='coerce')
    df['weekend'] = df['measured_date'].dt.day_name()
    df['is_weekend_or_not'] = df['weekend'].isin(['Saturday', 'Sunday']).astype(int)
    return df

#def missing_days_flag(df):
    #df['measured_date'] = pd.to_datetime(df['measured_date'], errors='coerce')
    #df = df.sort_values(['measured_date'])
    #df['missing_day'] = 0
    #for id in df['fitmri_id'].unique():
        # express the different id
        # for i in range(len()-1):
            #cur = df[i]
            #nxt = df[i+1]
            #if (nxt - cur) > 1:
                #flag = df[i+1, 'flag']

def save_results(df, filename='output/analysis_result.csv'):
    df.to_csv(filename, index=False)
    print("Results saved")

def main():
    df = load_csv()
    summarize_data(df)
    hourly_avg_df = hourly_average_steps(df)
    weekend_df = is_weekend(df)

    hourly_avg_df['measured_date'] = pd.to_datetime(hourly_avg_df['measured_date'], errors='coerce')
    weekend_df['measured_date'] = pd.to_datetime(weekend_df['measured_date'], errors='coerce')

    fianl_df = pd.merge(hourly_avg_df, weekend_df[['fitmri_id', 'measured_date', 'is_weekend_or_not']], on=['fitmri_id', 'measured_date'], how ='left')

    save_results(fianl_df)
    print('Final_df is done')

if __name__ == '__main__':
    main()