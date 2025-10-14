import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('output/analysis_result.csv')

def plot(df):
    plt.figure(figsize=(10, 5))

    # Plot a single id (change selected_id to the id you want)
    selected_id = 1
    patient_data = df[df['fitmri_id'] == selected_id]
    plt.plot(patient_data['measured_date'], patient_data['hourly_avg_steps']) ## draw it with date and avf steps

    plt.title('Hourly Average Steps (All Patients)')
    plt.xlabel('Date')
    plt.ylabel('Hourly Average Steps')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot(df)

def one_patient_plot(df, patient_id):

    plt.figure(figsize=(10, 5))
    patient_data = df[df['fitmri_id'] == patient_id]
    plt.plot(patient_data['measured_date'], patient_data['hourly_avg_steps'])

    plt.title('Hourly Average Steps (Selceted Patients)')
    plt.xlabel('Date')
    plt.ylabel('Hourly Average Steps')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

one_patient_plot(df, patient_id = 1) ### guys, select your patient id here :) hehehe