import pandas as pd


csv_file = r'C:\Users\Mensh\Desktop\divi\result.csv'


df = pd.read_csv(csv_file)


print(df.head())