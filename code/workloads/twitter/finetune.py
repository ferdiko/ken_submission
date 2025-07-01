# Based on:
# https://github.com/sarang0909/Explore-PyTorch/blob/master/Part2_Pytorch_Sentiment_Analysis.ipynb
#
import pandas as pd
import numpy as np
import torch
from sklearn import metrics
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification


# ================================================================================
# Configure
# ================================================================================
NUM_EPOCHS = 50
LEARNING_RATE = 0.01
TRAIN_FRAC = 0.8 # fraction of data to be used for training (rest for testing)
DATASET_PATH = "twitter_sentiment.csv"
SUBSET_SIZE = 1000 # Only use subset of data, set to -1 to use full
MODEL = "tiny" # ["tiny", "mini", "small", "medium", "base"]
RANDOM_SEED = 42
BATCH_SIZE = 128
CHECKPOINT_PATH = None # Don't checkpoint if None
# ================================================================================


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def eval_model(model, dl, out_path=None):
    """
    Evaluate the model, write predictions to file if out_path is not None.
    :param model: model
    :param dl: dataloader for test set
    :param out_path: Path of file for writing the predictions
    :return:
    """
    model.eval()

    # If requested, create file to write predictions to
    if out_path is not None:
        f = open(out_path, "w+")
        # f.write(f"pred,true\n")

    # predict test set batch-wise
    total_acc = 0.0
    num_batches = 0
    for x, x_att, y_label in dl:
        outputs = model(x.to(device), attention_mask=x_att.to(device))
        y_pred = torch.sigmoid(outputs[0]).detach().numpy()

        y_label = y_label.detach().numpy().astype(int)

        # if requested, write predictions to file (for simulator)
        if out_path is not None:
            for yp, yl in zip(y_pred, y_label):
                f.write(f"{yp[0]},{yl[0]}\n")

        # get binary class from predition
        threshold = 0.5
        y_pred = np.where(y_pred >= threshold, 1, 0).astype(int)

        total_acc += metrics.accuracy_score(y_label, y_pred)
        num_batches += 1

    # Compute average over batches
    total_acc /= num_batches
    print("Test accuracy: %.3f" % total_acc)
    model.train()


if __name__ == "__main__":
    # 0. Check if GPU available, set random seeds.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    np.random.seed(RANDOM_SEED) # sets seed for pandas (pandas uses np random seed by default)
    torch.manual_seed(RANDOM_SEED)

    # 1. Load model and tokenizer.
    model_full_names = {
        "tiny": "google/bert_uncased_L-2_H-128_A-2",
        "mini": "google/bert_uncased_L-4_H-256_A-4",
        "small": "google/bert_uncased_L-4_H-512_A-8",
        "medium": "google/bert_uncased_L-8_H-512_A-8",
        "base": "google/bert_uncased_L-12_H-768_A-12",
        # "large": "bert-large-uncased",
        # "distil": "models--distilbert-base-uncased"
    }
    model_name = model_full_names[MODEL]
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)
    tokenizer = BertTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2") # all BERTs use same tokenizer

    # 2. Load weights from checkpoint.
    # checkpoint = torch.load("checkpoints/casc_tiny_ll_1.pt")["model_state_dict"]
    # model.load_state_dict(checkpoint, strict=False)

    # 3. Done loading model.
    print("Done loading model.")
    # print(model)
    model = model.to(device)

    # 3. Load dataset.
    dataset = pd.read_csv(DATASET_PATH, encoding='latin-1', names=['target', 'id', 'date', 'query', 'username', 'text'])
    if SUBSET_SIZE > -1:
        dataset = dataset.sample(frac=SUBSET_SIZE/len(dataset))
    print("Done loading dataset.")

    # 4. Tokenize tweets and pad to equal length, attention mask: tell what is padding
    tokenized_reviews = dataset.text.apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
    max_len = max(map(len,tokenized_reviews))
    padded_reviews = np.array([ i+[0]*(max_len-len(i))  for i in tokenized_reviews])
    attention_masked_reviews = np.where(padded_reviews!=0,1,0)
    print("Done tokenizing.")

    X = torch.tensor(padded_reviews)
    X_att = torch.tensor(attention_masked_reviews)
    y = torch.tensor(dataset.target.values).float().reshape(-1, 1)
    y /= 4 # so probablilities

    # 5. Split into train / test set.
    cutoff = int(TRAIN_FRAC * X.shape[0])
    X_train = X[:cutoff, :]
    X_test = X[cutoff:, :]
    X_att_train = X_att[:cutoff, :]
    X_att_test = X_att[cutoff:, :]
    y_train = y[:cutoff, :]
    y_test = y[cutoff:, :]

    # 6. Create dataloaders.
    train_dataset = TensorDataset(X_train, X_att_train, y_train)
    test_dataset = TensorDataset(X_test, X_att_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 7. Init optimizer and loss function.
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    # loss_fn = torch.nn.CrossEntropyLoss()

    # 8. Train model
    print("Start training.")
    eval_model(model, test_dataloader)
    for _ in range(NUM_EPOCHS):
        for X_batch, X_att_batch, y_batch in tqdm(train_dataloader):
            # forward
            X_batch = X_batch.to(device)
            X_att_batch = X_att_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(X_batch, attention_mask=X_att_batch)

            # backward
            loss = loss_fn(y_pred[0], y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        eval_model(model, test_dataloader)

    # 9. Save checkpoint
    if CHECKPOINT_PATH is not None:
        torch.save(model.state_dict(), CHECKPOINT_PATH)
