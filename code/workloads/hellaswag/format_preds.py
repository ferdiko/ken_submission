import pandas as pd

# Define file paths
model1_path = "hellaswag_results/Meta-Llama-3.1-70B-Instruct.csv"
model2_path = "hellaswag_results/Meta-Llama-3.1-8B-Instruct.csv"

# Load the data without headers
df_70b = pd.read_csv(model1_path, header=None)
df_8b = pd.read_csv(model2_path, header=None)

# Check if the number of rows match before concatenating
if len(df_70b) == len(df_8b):
    # Extract the last column from the 8B file and add it to the 70B file
    df_70b['last_column_8b'] = df_8b.iloc[:, -1]
    
    # Save the modified 70B file without headers
    df_70b.to_csv(model1_path+"_new", index=False, header=False)
else:
    print("The files have a different number of rows and cannot be merged directly.")
