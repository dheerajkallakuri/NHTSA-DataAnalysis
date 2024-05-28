import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('YOUR_DATA_FILE.csv') #ADS/ADAS

# Replace NaN (empty) values with True and non-empty values with False in a specific column
column_name = 'COLUMN_NAME'  # Replace with the actual column name
empty_cells = df[column_name].isna()

# Now, the 'empty_cells' Series contains True for empty cells and False for non-empty cells

# To get the count of empty cells in that column, you can use the sum() function
empty_count = empty_cells.sum()

print(f"Number of empty cells in column '{column_name}': {empty_count}")