import pandas as pd

"""
Simple script to read a Feather file and print its basic info plus the first 10 rows.
"""

def main():
    # Hard-coded Feather file path
    file = r"E:\refav\RefAV\av2_sm_downloads\scenario_mining_val_annotations.feather"

    # Read the Feather file into a DataFrame
    try:
        df = pd.read_feather(file)
    except Exception as e:
        print(f"Error reading Feather file: {e}")
        return

    # Print basic information
    print(f"File: {file}")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("Column dtypes:")
    print(df.dtypes)
    print()

    # Always show the first 10 rows
    head_n = 40
    print(f"Showing first {head_n} rows:")
    print(df.head(head_n).to_string(index=False))

if __name__ == "__main__":
    main()