from datasets import load_dataset
import sentencepiece as spm
import os
import torch
import numpy as np
import random

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from models.utils import pad_sequences 

# https://huggingface.co/datasets/tweet_eval
# https://huggingface.co/datasets/super_glue


def dataset_importer(dataset_name, vocab_size, max_length = 512, batch_size = 32):

  #load dataset
  if dataset_name == "imdb":
    dataset = load_dataset("imdb")
    train_data = dataset['train']
    test_data = dataset['test']

    text_train = train_data.to_dict()["text"]
    label_train = train_data.to_dict()["label"]

    text_test = test_data.to_dict()["text"]
    label_test = test_data.to_dict()["label"]

    N_LABELS = len(set(label_train))

  elif dataset_name == "Amazon": #NEEDS FIXING: LABELS ARE 0 through 4 but only 0 and 4 are used...
    dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)

    text_full = [tit + " " + txt for tit, txt in zip(dataset["full"]["title"], dataset["full"]["text"])]
    label_full = dataset["full"]["rating"]
    label_full = (torch.Tensor(label_full) - 1).tolist() #rescaling labels to 0-4

    text_train = text_full[:len(label_full)//10 * 9]
    label_train = label_full[:len(label_full)//10 * 9]

    text_test = text_full[len(label_full)//10 * 9 + 1: ]
    label_test = label_full[len(label_full)//10 * 9 + 1: ]

    N_LABELS = len(set(label_full))

  elif dataset_name == "sst2":
    dataset = load_dataset("sst2")

    train_data = dataset['train']
    val_data = dataset['validation']
    # test_data = dataset['test'] #sti idioti non hanno classificato il test set

    text_train = train_data.to_dict()["sentence"]
    label_train = train_data.to_dict()["label"]

    text_test = val_data.to_dict()["sentence"]
    label_test = val_data.to_dict()["label"]

    N_LABELS = len(set(label_train))

  elif dataset_name == "sst5":
    dataset = load_dataset("SetFit/sst5")

    train_data = dataset['train']
    val_data = dataset['validation']
    test_data = dataset['test']

    text_train = train_data.to_dict()["text"]
    label_train = train_data.to_dict()["label"]

    text_test = test_data.to_dict()["text"] +  val_data.to_dict()["text"]
    label_test = test_data.to_dict()["label"] + val_data.to_dict()["label"]

    N_LABELS = len(set(label_train))

  elif dataset_name == "twitter":
    dataset = load_dataset("tweet_eval", "emoji")
    train_data = dataset['train']
    val_data = dataset['validation']
    test_data = dataset['test']

    text_train = train_data["text"] + val_data["text"]
    label_train = train_data["label"] + val_data["label"]

    text_test = test_data["text"]
    label_test = test_data["label"]

    N_LABELS = len(set(label_train))


  assert len(text_train) == len(label_train)
  assert len(text_test) == len(label_test)
  assert(N_LABELS > 1)

  if not os.path.exists("./train_ds_" + dataset_name + ".txt"):
    #reduce dataset for sentencepiece training
    try:
      idxs = random.sample(range(len(text_train)), 10000)
      aux = [text_train[i] for i in idxs]
    except:
      aux = text_train
    #save file of dataset for tokenizer
    filename = "./train_ds_" + dataset_name + ".txt"
    with open(filename, 'w') as f:
      for s in aux:
        f.write(s)

  if not os.path.exists("./m_" + dataset_name + ".model"):
    #Train tokenizer
    spm.SentencePieceTrainer.train(input="./train_ds_" + dataset_name + ".txt", model_prefix="m_" + dataset_name, max_sentence_length = 100000000 ,vocab_size=vocab_size)


  #Load tokenizer
  sp = spm.SentencePieceProcessor(model_file="m_" + dataset_name + ".model")

  print(f'Dictionary size {sp.get_piece_size()}')
  # print(f'Vocabulary: {[sp.id_to_piece(id) for id in range(sp.get_piece_size())]}')
  print(f'Encoding results:  {sp.encode("this is a phrase that could be commonly found", out_type=str)} -> {sp.encode("this is a phrase that could be commonly found")}')

  train_tokens = list(map(lambda t: [1] + sp.encode(t)[:max_length - 2] + [2], text_train))
  test_tokens = list(map(lambda t: [1] + sp.encode(t)[:max_length - 2] + [2], text_test))

  train_tokens_ids = pad_sequences(train_tokens, maxlen = max_length, truncating="post", padding="post", dtype="int")
  test_tokens_ids = pad_sequences(test_tokens, maxlen = max_length, truncating="post", padding="post", dtype="int")

  train_masks_tensor = (torch.tensor(train_tokens_ids) > 0).float()
  test_masks_tensor = (torch.tensor(test_tokens_ids) > 0).float()



  train_tokens_tensor = torch.tensor(train_tokens_ids)
  #train_y_tensor = torch.tensor(np.array(label_train).reshape(-1, 1)).long()
  train_y_tensor = torch.tensor(np.array(label_train)).long()

  test_tokens_tensor = torch.tensor(test_tokens_ids)
  #test_y_tensor = torch.tensor(np.array(label_test).reshape(-1, 1)).long()
  test_y_tensor = torch.tensor(np.array(label_test)).long()

  print(N_LABELS)
  print(train_tokens_tensor[:10])
  print(train_y_tensor[:10])




  train_dataset = TensorDataset(train_tokens_tensor, train_masks_tensor, train_y_tensor)
  train_sampler = RandomSampler(train_dataset)
  train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

  test_dataset = TensorDataset(test_tokens_tensor, test_masks_tensor, test_y_tensor)
  test_sampler = SequentialSampler(test_dataset)
  test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)


  return train_dataloader, test_dataloader, N_LABELS, label_test