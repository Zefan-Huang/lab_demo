import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('output/analysis_result.csv')

def plot(df):
    plt.figure(figsize=(10, 5))

    for id in df['fitmri_id'].unique():   #id = 1,2,3 ... 26
        patient_data = df[df['fitmri_id'] == id]    ## whole row of cur id
        plt.plot(patient_data['measured_date'], patient_data['hourly_avg_steps']) ## draw it with date and avf steps

    plt.title('Hourly Average Steps (All Patients)')
    plt.xlabel('Date')
    plt.ylabel('Hourly Average Steps')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot(df)