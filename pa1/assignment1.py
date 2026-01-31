import gzip
import json
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
#wow ALS was complete ass now i write tf-idf and stuff

def readGz(path):
  for l in gzip.open(path, 'rt'):
    yield eval(l)

def readCSV(path):
  f = gzip.open(path, 'rt')
  f.readline()
  for l in f:
    yield l.strip().split(',')

### Would-read baseline: just rank which books are popular and which are not, and return '1' if a book is among the top-ranked

userBooks = defaultdict(set)
bookUsers = defaultdict(set)
bookCounts = defaultdict(int)

for u, b, _ in readCSV("train_Interactions.csv.gz"):
    userBooks[u].add(b)
    bookUsers[b].add(u)
    bookCounts[b] += 1

totalReads = sum(bookCounts.values())
mostPopular = sorted(bookCounts.items(), key=lambda x: x[1], reverse=True)

def popularBooksThreshold(mostPopular, totalReads, threshold=0.6):
    topBooks = set()
    count = 0
    limit = totalReads * threshold
    for b, c in mostPopular:
        topBooks.add(b)
        count += c
        if count > limit:
            break
    return topBooks

topBooks = popularBooksThreshold(mostPopular, totalReads, threshold=0.6)

def jaccardSimilarity(u, b):
    maxSim = 0
    users_b = bookUsers.get(b, set())
    for bprime in userBooks.get(u, set()):
        if bprime == b: continue
        users_bp = bookUsers.get(bprime, set())
        if not users_b or not users_bp:
            continue
        sim = len(users_b.intersection(users_bp)) / len(users_b.union(users_bp))
        if sim > maxSim:
            maxSim = sim
    return maxSim


with open("predictions_Read.csv", 'w') as f:
    for l in open("pairs_Read.csv"):
        if l.startswith("userID"):
            f.write(l)
            continue
        u, b = l.strip().split(',')

        sim = jaccardSimilarity(u, b)
        pred = 1 if b in topBooks or sim > 0.013 or len(bookUsers.get(b, [])) > 40 else 0
        f.write(f"{u},{b},{pred}\n")


### Category prediction baseline: Just consider some of the most common words from each category

train_data = list(readGz("train_Category.json.gz"))
test_data = list(readGz("test_Category.json.gz"))

train_texts = [d['review_text'] for d in train_data]
train_labels = [d['genreID'] for d in train_data]
test_texts = [d['review_text'] for d in test_data]

class ReviewDataset(Dataset):
    def __init__(self, texts, labels=None, max_len=128):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_dataset = ReviewDataset(train_texts, train_labels)
test_dataset = ReviewDataset(test_texts)

num_classes = 5
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=num_classes
)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    num_train_epochs=2,
    save_steps=500,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()

preds = trainer.predict(test_dataset)
pred_labels = preds.predictions.argmax(-1)

with open("predictions_Category.csv", 'w') as f:
    f.write("userID,reviewID,prediction\n")
    for d, p in zip(test_data, pred_labels):
        f.write(f"{d['user_id']},{d['review_id']},{p}\n")

### Rating baseline: compute averages for each user, or return the global average if we've never seen the user before

print("Starting Rating Prediction...")
allRatings = []
userRatings = defaultdict(list)
bookRatings = defaultdict(list)

for u, b, r in readCSV("train_Interactions.csv.gz"):
    r = int(r)
    allRatings.append(r)
    userRatings[u].append(r)
    bookRatings[b].append(r)

globalAvg = sum(allRatings)/len(allRatings)

with open("predictions_Rating.csv",'w') as f:
    for l in open("pairs_Rating.csv"):
        if l.startswith("userID"):
            f.write(l)
            continue
        u,b = l.strip().split(',')
        r = globalAvg
        if u in userRatings:
            user_avg = sum(userRatings[u])/len(userRatings[u])
            r += 0.5 * (user_avg - globalAvg)
        if b in bookRatings:
            book_avg = sum(bookRatings[b])/len(bookRatings[b])
            r += 0.5 * (book_avg - globalAvg)
        r = max(1, min(5, r))
        f.write(f"{u},{b},{r}\n")


