import pandas as pd



df = pd.read_csv('window_inverter_11_14.csv')


df['Timestamp'] = pd.to_datetime(df['Timestamp'])


specific_date = '2023-11-14'
start_hour = 11
end_hour = 14


column_to_modify = 'DC_Voltage'
column_to_modify_1='DC_Current'


mask = (df['Timestamp'].dt.date == pd.to_datetime(specific_date).date()) & \
       (df['Timestamp'].dt.hour >= start_hour) & \
       (df['Timestamp'].dt.hour < end_hour)


df.loc[mask, column_to_modify] = 1.5*df.loc[mask, column_to_modify]
df.loc[mask, column_to_modify_1] = 1.5*df.loc[mask, column_to_modify_1]
print(df[0:50])

df.to_csv('attack_14.csv', index=False)






