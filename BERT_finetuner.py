from colors import ATTRIBUTES, FOREGROUND_COLORS, RESET

import torch
import sentencepiece as spm
from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from tqdm import tqdm


import BERTs.BERT_Eff_Enc_Gray as BERT
from utils import model_size

device = 'cuda' if torch.cuda.is_available() else 'cpu'

VOCAB_SIZE = 512*8
BATCH_SIZE = 32
SENTENCE_LEN = 512


tokens = {
    "pad":0,
    "bos":1,
    "eos":2,
    "unk":3,
    "mask":4
}


 
print(f"{ATTRIBUTES['Bold']}Importing sentencepiece processor...{RESET}")
#import sentencepiece model
sp = spm.SentencePieceProcessor()
sp.load("./stpiece/m_orca_dictsz_4096.model")


print(f"{ATTRIBUTES['Bold']}Loading dataset...{RESET}")
dataset = load_dataset("trec")#imdb, ag_news, trec
train_data = dataset['train']
test_data = dataset['test']

text_train = train_data.to_dict()["text"]
#label_train = train_data.to_dict()["label"]
label_train = train_data.to_dict()["coarse_label"]


text_test = test_data.to_dict()["text"]
#label_test = test_data.to_dict()["label"]
label_test = test_data.to_dict()["coarse_label"]

N_LABELS = len(set(label_train))

print(f"{ATTRIBUTES['Bold']}Building dataset...{RESET}")
train_tokens = list(map(lambda t: [tokens["bos"]] + sp.encode(t)[:SENTENCE_LEN - 2] + [tokens["eos"]], text_train))
test_tokens = list(map(lambda t: [tokens["bos"]] + sp.encode(t)[:SENTENCE_LEN - 2] + [tokens["eos"]], text_test))

train_tokens_ids = [train_tokens[i] +  [0] * (SENTENCE_LEN - len(train_tokens[i])) for i in range(len(train_tokens))]
test_tokens_ids = [test_tokens[i] +  [0] * (SENTENCE_LEN - len(test_tokens[i])) for i in range(len(test_tokens))]

train_tokens_tensor = torch.tensor(train_tokens_ids)
train_y_tensor = torch.tensor(np.array(label_train)).long()

test_tokens_tensor = torch.tensor(test_tokens_ids)
test_y_tensor = torch.tensor(np.array(label_test)).long()


train_dataset = TensorDataset(train_tokens_tensor, train_y_tensor)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE, pin_memory=True)

test_dataset = TensorDataset(test_tokens_tensor, test_y_tensor)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE, pin_memory=True)


print(f"{ATTRIBUTES['Bold']}Loading model...{RESET}")

bert_model = BERT.BERT(
  vocab_size=VOCAB_SIZE,
  d_model=128,
  n_layers=4,
  heads=8,
  sentence_length=SENTENCE_LEN,
  dropout=0.1
)

bert_model.load_state_dict(torch.load("./models/bert_model_gray_wiki_128_4_4096_20240423_153737.pth"))

class BERT_classifier(torch.nn.Module):
    def __init__(self, bert_model, num_classes, freeze_bert = True):
        super().__init__()
        self.model = bert_model
        if freeze_bert:
            for param in self.model.parameters():
                param.requires_grad = False
        #concatenation of two linear layers 
        self.ff = torch.nn.Linear(self.model.d_model, self.model.d_model)
        self.classifier = torch.nn.Linear(self.model.d_model, num_classes)
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        
    def forward(self, x):
        mask = (x != 0).int()
        x = self.model(x, mask)
        x = self.ff(torch.nn.functional.tanh(x[:, 0]))
        return self.softmax(self.classifier(torch.nn.functional.tanh(x)))
    

#initialize the model
bert_classifier = BERT_classifier(bert_model=bert_model, num_classes=N_LABELS, freeze_bert=False).to(device)   
print(model_size(bert_classifier))


print(f"{ATTRIBUTES['Bold']}Training...{RESET}")
print(f"{FOREGROUND_COLORS['Green']}", end="")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

criterion = torch.nn.CrossEntropyLoss()
criterion.to(device);

optimizer = torch.optim.Adam(bert_classifier.parameters(), lr=10e-4)

for epoch in range(5):  
    bert_classifier.train()
    train_loss_avg = 0
    train_loss = 0
    tqdm_train_loader = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", leave=False)

    for step_num, batch_data in enumerate(tqdm_train_loader):

        token_ids, labels = tuple(t.to(device) for t in batch_data)

        logits = bert_classifier(token_ids)

        batch_loss = criterion(logits, labels)
        train_loss += batch_loss.item()
        train_loss_avg += batch_loss.item()

        bert_classifier.zero_grad()
        batch_loss.backward()

		#torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        log_step = 50
        if step_num % log_step == (log_step - 1):
            tqdm_train_loader.set_postfix(loss = train_loss / log_step)
            train_loss = 0
    
    print(f"Epoch {epoch+1} completed. Loss: {train_loss_avg / step_num}")

print(f"{RESET}")

print(f"{ATTRIBUTES['Bold']}Evaluation...{RESET}")
print(f"{FOREGROUND_COLORS['BrightBlue']}", end="")
criterion = torch.nn.CrossEntropyLoss()
criterion.to(device);

bert_classifier.eval()
predicted = torch.Tensor([])
all_logits = torch.Tensor([])

tqdm_test_loader = tqdm(test_dataloader, desc=f"Evaluation: ", leave=False)
	
avg_loss = 0
with torch.no_grad():
	for step_num, batch_data in enumerate(tqdm_test_loader):

		token_ids, labels = tuple(t.to(device) for t in batch_data)

		logits = bert_classifier(token_ids)
		loss = criterion(logits, labels)
		avg_loss += loss.item()
		numpy_logits = logits.cpu().detach()
		predicted = torch.cat((predicted, torch.argmax(numpy_logits, dim = 1)))
		all_logits  = torch.cat((all_logits, numpy_logits))


print("average loss: ", avg_loss / len(test_dataloader))
print("Accuracy: ", (predicted == test_y_tensor).sum().item() / len(test_y_tensor))

print(f"{RESET}")