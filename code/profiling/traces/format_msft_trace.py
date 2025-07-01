import pandas as pd


if __name__ == "__main__":

    out_file = "msft_app1.csv"

    # Read file.
    # with open("msft_unformatted.txt", "r") as f:
    #     data = f.read()

    # Read the data into a DataFrame.
    df = pd.read_csv("msft_unformatted.txt")

    # Filter out timestamps of one app.
    df = df[df['app'] == '7b2c43a2bc30f6bb438074df88b603d2cb982d3e7961de05270735055950a568']

    # Convert to timestamps and write to file,
    df['timestamp_diff'] = df['end_timestamp'] - df['duration']
    df['timestamp_diff'].to_csv(out_file, index=False, header=False)