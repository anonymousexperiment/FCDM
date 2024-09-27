import pandas as pd

# Define the list of CSV file paths
csv_files = [
    '/root/DiME-main/celebA_dataset/Dataprocess/0611celebA_original.csv',
    '/root/DiME-main/celebA_dataset/Dataprocess/0613celebA_original.csv',
    '/root/DiME-main/celebA_dataset/Dataprocess/0614celebA_original.csv',
    '/root/DiME-main/celebA_dataset/Dataprocess/0615celebA_original.csv',
    '/root/DiME-main/celebA_dataset/Dataprocess/0616celebA_original.csv',
    '/root/DiME-main/celebA_dataset/Dataprocess/0617celebA_original.csv',
    '/root/DiME-main/celebA_dataset/Dataprocess/0611celebA_cf.csv',
    '/root/DiME-main/celebA_dataset/Dataprocess/0613celebA_cf.csv',
    '/root/DiME-main/celebA_dataset/Dataprocess/0614celebA_cf.csv',
    '/root/DiME-main/celebA_dataset/Dataprocess/0615celebA_cf.csv',
    '/root/DiME-main/celebA_dataset/Dataprocess/0616celebA_cf.csv',
    '/root/DiME-main/celebA_dataset/Dataprocess/0617celebA_cf.csv'
]

total_rows = 0

# Loop through each CSV file
for csv_file in csv_files:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    # Get the number of rows in the DataFrame
    num_rows = len(df)
    # Print the number of rows for this CSV file
    print(f"{csv_file} has {num_rows} rows")
    # Add the number of rows to the total
    total_rows += num_rows

# Print the total number of rows
print(f"Total number of rows: {total_rows}")
