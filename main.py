from colors import ATTRIBUTES, FOREGROUND_COLORS, RESET


import BERTs.BERT_Eff_Enc_Gray as BERT
from BERTs.LLM import BERTLM
from trainer import BERTTrainer
import BERTdataset
from utils import model_size, train_sp

from datasets import load_dataset


import torch
import tqdm

from torch.utils.data import DataLoader
import datetime


device = 'cuda' if torch.cuda.is_available() else 'cpu'

VOCAB_SIZE = 512*8
BATCH_SIZE = 256
SENTENCE_LEN = 512


tokens = {
    "pad":0,
    "bos":1,
    "eos":2,
    "unk":3,
    "mask":4
}
    
print(f"{ATTRIBUTES['Bold']}Loading dataset...{RESET}")
dataset = load_dataset("Open-Orca/OpenOrca", cache_dir="./orca_madonna", trust_remote_code=True, split=['train[:1%]', 'train[80%:81%]', 'train[90%:91%]'])

#dataset is composed of a dictionary containig train, validation and test
#each of them is a list of dictionaries containing the following keys: ['id', 'system_prompt', 'question', 'response']
train_data = dataset[0]
validation_data = dataset[1]
test_data = dataset[2]

sp = train_sp(train_data, VOCAB_SIZE, tokens)

print(f"{ATTRIBUTES['Bold']}Tokenizing data...{RESET}")
print("\tTrain dataset")
train_data_tokenized = list(zip(sp.encode(train_data["question"]), sp.encode(train_data["response"])))
print("\tValidation dataset")
val_data_tokenized = list(zip(sp.encode(validation_data["question"]), sp.encode(validation_data["response"])))
print("\tTest dataset")
test_data_tokenized = list(zip(sp.encode(test_data["question"]), sp.encode(test_data["response"])))
#average length of tokenized sentences
print(f"Average length of tokenized sentences: {sum([len(x[0]) + len(x[1]) for x in train_data_tokenized])/len(train_data_tokenized)}")

#################################################################################################################################
print(f"{ATTRIBUTES['Bold']}Building dataset...{RESET}")

train_data = BERTdataset.BERTDataset(train_data_tokenized, special_tokens=tokens, seq_len=SENTENCE_LEN)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

valid_data = BERTdataset.BERTDataset(val_data_tokenized, special_tokens=tokens, seq_len=SENTENCE_LEN)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

#################################################################################################################################
print(f"{ATTRIBUTES['Bold']}Building BERT model...{RESET}")

bert_model = BERT.BERT(
  vocab_size=VOCAB_SIZE,
  d_model=128,
  n_layers=6,
  heads=8,
  sentence_length=SENTENCE_LEN,
  dropout=0.1
)

bert_lm = BERTLM(bert_model, VOCAB_SIZE)
bert_lm = bert_lm.to(device)

print("BERT model parameters without classifier:")
#print all model parameters with names
for name, param in bert_model.named_parameters():
	print(f"{name}: {param.nelement()}")
#print the model size
print(model_size(bert_model))


#################################################################################################################################
print(f"{ATTRIBUTES['Bold']}Training BERT model...{RESET}")
bert_trainer = BERTTrainer(bert_lm, train_loader, valid_loader, device=device, log_freq=len(train_loader)//5)
epochs = 5

for epoch in range(epochs):
  print(f"{FOREGROUND_COLORS['Green']}", end="")
  bert_trainer.train(epoch)
  print(f"{FOREGROUND_COLORS['BrightBlue']}", end="")
  bert_trainer.test(epoch)
  print(f"{RESET}")


#################################################################################################################################
print(f"{ATTRIBUTES['Bold']}Saving BERT model...{RESET}")
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
torch.save(bert_lm.state_dict(), "./models/bert_lm_" + str(bert_lm.bert.d_model) + "_" + str(bert_lm.bert.n_layers) + "_" + str(VOCAB_SIZE) + "_" + timestamp + ".pth")
torch.save(bert_model.state_dict(), "./models/bert_model_" + str(bert_model.d_model) + "_" + str(bert_model.n_layers) + "_" + str(VOCAB_SIZE) + "_" + timestamp + ".pth")