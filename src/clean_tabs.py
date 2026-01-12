import csv
import sys
import pandas as pd

# Allow very large fields in CSV
csv.field_size_limit(sys.maxsize)

# Define paths
input_path = '/Users/dventr/Downloads/aktuelle_parteiprogramme_DACH.tsv'
output_path = 'cleaned_file.tsv'

# Expected number of columns
num_columns = 4

with open(input_path, 'r', encoding='utf-8') as infile, \
     open(output_path, 'w', encoding='utf-8', newline='') as outfile:
    
    reader = csv.reader(infile, delimiter='\t')
    writer = csv.writer(outfile, delimiter='\t')

    for row in reader:
        # Remove any inner tabs from fields
        row = [field.replace('\t', ' ') for field in row]

        if len(row) > num_columns:
            # Keep first 3 columns, merge the rest into the 4th
            cleaned_row = row[:3] + [' '.join(row[3:])]
        else:
            # Pad missing columns if too short
            cleaned_row = row + [''] * (num_columns - len(row))

        writer.writerow(cleaned_row)

df = pd.read_csv('cleaned_file.tsv', sep='\t')
print(df.head())