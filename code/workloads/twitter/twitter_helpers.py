import os
import csv
import random


def get_tweets(num_tweets):
    random.seed(2)

    dataset_path = os.path.join(os.path.dirname(__file__), "../../profiling/traces/twitter_sentiment.csv")
    with open(dataset_path, "r", encoding='latin-1') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')

        # TODO: Check that you only use the test set
        tweets = [r[5] for r in reader]
        random.shuffle(tweets)

        return tweets[:num_tweets]


if __name__ == "__main__":
    print(get_tweets(40))
