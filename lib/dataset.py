from datasets import load_dataset
import sentencepiece as spm
import os
import torch
import numpy as np
import random

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


from lib.utils import pad_sequences, train_sp

# https://huggingface.co/datasets/tweet_eval
# https://huggingface.co/datasets/super_glue


# https://huggingface.co/datasets/blog_authorship_corpus #WE COULD USE IT WITH DIFFERENT LABELS
# https://huggingface.co/datasets/limit #COULD BE INTERESTING TO MAKE IT ALSO CLASSIFY THE TOKEN THAT IS MOVING (much harder)

def dataset_importer(dataset_name, vocab_size, max_length = 512, batch_size = 32, custom_sentencepiece = None):

	#load dataset
	if dataset_name == "imdb":
		dataset = load_dataset("imdb", cache_dir="./datasets")
		train_data = dataset['train']
		test_data = dataset['test']

		text_train = train_data.to_dict()["text"]
		label_train = train_data.to_dict()["label"]

		text_test = test_data.to_dict()["text"]
		label_test = test_data.to_dict()["label"]
		#count the occurrences of each label in train set
		_, counts = np.unique(np.array(label_train), return_counts=True)

		LABELS = counts

	elif dataset_name == "Amazon": #NEEDS FIXING: LABELS ARE 0 through 4 but only 0 and 4 are used...
		dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True, cache_dir="./datasets")

		text_full = [tit + " " + txt for tit, txt in zip(dataset["full"]["title"], dataset["full"]["text"])]
		label_full = dataset["full"]["rating"]
		label_full = (torch.Tensor(label_full) - 1).tolist() #rescaling labels to 0-4

		text_train = text_full[:len(label_full)//10 * 9]
		label_train = label_full[:len(label_full)//10 * 9]

		text_test = text_full[len(label_full)//10 * 9 + 1: ]
		label_test = label_full[len(label_full)//10 * 9 + 1: ]

		#count the occurrences of each label in train set
		_, counts = np.unique(np.array(label_train), return_counts=True)

		LABELS = counts

	elif dataset_name == "sst2":
		dataset = load_dataset("sst2", cache_dir="./datasets")

		train_data = dataset['train']
		val_data = dataset['validation']
		# test_data = dataset['test'] #sti idioti non hanno classificato il test set

		text_train = train_data.to_dict()["sentence"]
		label_train = train_data.to_dict()["label"]

		text_test = val_data.to_dict()["sentence"]
		label_test = val_data.to_dict()["label"]

		#count the occurrences of each label in train set
		_, counts = np.unique(np.array(label_train), return_counts=True)

		LABELS = counts

	elif dataset_name == "twitter":
		dataset = load_dataset("tweet_eval", "emoji", cache_dir="./datasets")
		train_data = dataset['train']
		val_data = dataset['validation']
		test_data = dataset['test']

		text_train = train_data["text"] + val_data["text"]
		label_train = train_data["label"] + val_data["label"]

		text_test = test_data["text"]
		label_test = test_data["label"]

		#count the occurrences of each label in train set
		_, counts = np.unique(np.array(label_train), return_counts=True)

		LABELS = counts

	elif dataset_name == "race":
		dataset = load_dataset("ehovy/race", "middle", cache_dir="./datasets")

		train_data = dataset['train']
		# val_data = dataset['validation']
		test_data = dataset['test']

		text_train = [art + "\nThe question is: " + que + "\nThe possible answers are:\nA:" + opt[0] + "\nB:" + opt[1] + "\nC:" + opt[2] + "\nD:" + opt[3] for art, que, opt in zip(train_data["article"], train_data["question"], train_data["options"])]
		label_train = [0 if ans == "A" else 1 if ans == "B" else 2 if ans == "C" else 3 for ans in train_data["answer"]]
		
		text_test = [art + "\nThe question is: " + que + "\nThe possible answers are:\nA:" + opt[0] + "\nB:" + opt[1] + "\nC:" + opt[2] + "\nD:" + opt[3] for art, que, opt in zip(test_data["article"], test_data["question"], test_data["options"])]
		label_test = [0 if ans == "A" else 1 if ans == "B" else 2 if ans == "C" else 3 for ans in test_data["answer"]]

		#count the occurrences of each label in train set
		_, counts = np.unique(np.array(label_train), return_counts=True)

		LABELS = counts

	elif dataset_name == "yelp":
		dataset = load_dataset("yelp_review_full", cache_dir="./datasets")

		train_data = dataset['train']
		test_data = dataset['test']

		text_train = train_data.to_dict()["text"]
		label_train = train_data.to_dict()["label"]

		text_test = test_data.to_dict()["text"]
		label_test = test_data.to_dict()["label"]

		#count the occurrences of each label in train set
		_, counts = np.unique(np.array(label_train), return_counts=True)

		LABELS = counts

	elif dataset_name == "news":
		dataset = load_dataset("ag_news", cache_dir="./datasets")
		
		train_data = dataset['train']
		test_data = dataset['test']

		text_train = train_data.to_dict()["text"]
		label_train = train_data.to_dict()["label"]

		text_test = test_data.to_dict()["text"]
		label_test = test_data.to_dict()["label"]

		#count the occurrences of each label in train set
		_, counts = np.unique(np.array(label_train), return_counts=True)

		LABELS = counts

	elif dataset_name == "trec_coarse":
		dataset = load_dataset("trec", cache_dir="./datasets")

		train_data = dataset['train']
		test_data = dataset['test']

		text_train = train_data.to_dict()["text"]
		label_train = train_data.to_dict()["coarse_label"]

		text_test = test_data.to_dict()["text"]
		label_test = test_data.to_dict()["coarse_label"]

		#count the occurrences of each label in train set
		_, counts = np.unique(np.array(label_train), return_counts=True)

		LABELS = counts

	elif dataset_name == "bull":
		#read csv file and import data
		
		import csv

		data_dict = {}
		text = []
		label = []

		#check if the file exists 
		if not os.path.exists('./datasets/cyberbullying_tweets.csv'):
			raise ValueError("Dataset not found") 

		with open('./datasets/cyberbullying_tweets.csv', mode ='r')as file:
		
			reader = csv.reader(file)		
			next(reader)  # Skip the header row
			for row in reader:
				text.append(row[0])
				label.append(row[1])
		# Create a mapping of labels to integers
		label_mapping = {label: i for i, label in enumerate(set(label))}
		print(label_mapping)

		# Transform the labels to integers
		label = [label_mapping[l] for l in label]

		#shuffle text and label in the same way
		combined = list(zip(text, label))
		random.shuffle(combined)
		text[:], label[:] = zip(*combined)
		
		text_train = text[:len(label)//10 * 9]
		label_train = label[:len(label)//10 * 9]

		text_test = text[len(label)//10 * 9 + 1: ]
		label_test = label[len(label)//10 * 9 + 1: ]

		#count the occurrences of each label in train set
		_, counts = np.unique(np.array(label_train), return_counts=True)

		LABELS = counts
	elif dataset_name == "limit":
		dataset = load_dataset("limit", cache_dir="./datasets")

		train_data = dataset['train']
		test_data = dataset['test']

		text_train = train_data.to_dict()["sentence"]
		label_train = train_data.to_dict()["motion"]
		#convert label from "yes" and "no" to 1 and o
		label_train = [1 if l == "yes" else 0 for l in label_train]

		text_test = test_data.to_dict()["sentence"]
		label_test = test_data.to_dict()["motion"]
		label_test = [1 if l == "yes" else 0 for l in label_test]

		#count the occurrences of each label in train set
		_, counts = np.unique(np.array(label_train), return_counts=True)

		LABELS = counts

	elif dataset_name == "dbpedia":
		dataset = load_dataset("fancyzhx/dbpedia_14", cache_dir="./datasets")

		train_data = dataset['train']
		test_data = dataset['test']

		text_train = train_data.to_dict()["content"]
		label_train = train_data.to_dict()["label"]

		text_test = test_data.to_dict()["content"]
		label_test = test_data.to_dict()["label"]

		#count the occurrences of each label in train set
		_, counts = np.unique(np.array(label_train), return_counts=True)

		LABELS = counts

	elif dataset_name == "nlu":
		dataset = load_dataset("nlu_evaluation_data", cache_dir="./datasets")

		train_data = dataset['train']

		text = train_data.to_dict()["text"]
		label = train_data.to_dict()["scenario"]

		label_mapping = {label: i for i, label in enumerate(set(label))}
		print(label_mapping)

		# Transform the labels to integers
		label = [label_mapping[l] for l in label]

		combined = list(zip(text, label))
		random.shuffle(combined)
		text[:], label[:] = zip(*combined)
		
		text_train = text[:len(label)//10 * 9]
		label_train = label[:len(label)//10 * 9]

		text_test = text[len(label)//10 * 9 + 1: ]
		label_test = label[len(label)//10 * 9 + 1: ]

		#count the occurrences of each label in train set
		_, counts = np.unique(np.array(label_train), return_counts=True)

		LABELS = counts

	elif dataset_name == "snips":
		dataset = load_dataset("benayas/snips", cache_dir="./datasets")

		train_data = dataset['train']
		test_data = dataset['test']

		text_train = train_data.to_dict()["text"]
		label_train = train_data.to_dict()["category"]

		text_test = test_data.to_dict()["text"]
		label_test = test_data.to_dict()["category"]

		label_mapping = {label: i for i, label in enumerate(set(label_train))}
		print(label_mapping)

		# Transform the labels to integers
		label_train = [label_mapping[l] for l in label_train]
		label_test = [label_mapping[l] for l in label_test]

		#count the occurrences of each label in train set
		_, counts = np.unique(np.array(label_train), return_counts=True)

		LABELS = counts

	elif dataset_name == "multi_nli":
		dataset = load_dataset("nyu-mll/multi_nli", cache_dir="./datasets")

		train_data = dataset['train']
		test_data = dataset['validation_matched']

		premise_train = train_data.to_dict()["premise"]
		consequence_train = train_data.to_dict()["hypothesis"]

		text_train = [premise + "\n" + consequence for premise, consequence in zip(premise_train, consequence_train)]
		label_train = train_data.to_dict()["label"]

		premise_test = test_data.to_dict()["premise"]
		consequence_test = test_data.to_dict()["hypothesis"]
		text_test = [premise + "\n" + consequence for premise, consequence in zip(premise_test, consequence_test)]
		label_test = test_data.to_dict()["label"]

		#count the occurrences of each label in train set
		_, counts = np.unique(np.array(label_train), return_counts=True)

		LABELS = counts

	elif dataset_name == "blog":
		dataset = load_dataset("blog_authorship_corpus", cache_dir="./datasets")

		train_data = dataset['train']
		test_data = dataset['validation']

		text_train = train_data.to_dict()["text"]
		label_train = train_data.to_dict()["job"]

		text_test = test_data.to_dict()["text"]
		label_test = test_data.to_dict()["job"]

		label_mapping = {label: i for i, label in enumerate(set(label_train))}
		print(label_mapping)

		# Transform the labels to integers
		label_train = [label_mapping[l] for l in label_train]
		label_test = [label_mapping[l] for l in label_test]

		#count the occurrences of each label in train set
		_, counts = np.unique(np.array(label_train), return_counts=True)

		LABELS = counts

	else:
		raise ValueError("Dataset not found")

	assert len(text_train) == len(label_train)
	assert len(text_test) == len(label_test)
	assert(len(LABELS) > 1)

	
 
	sp = train_sp(text_train, vocab_size, dataset_name)

	
	train_tokens = sp.encode(text_train)
	test_tokens = sp.encode(text_test)
	print(f"The average token length is {np.mean(list(map(len, train_tokens)))}")
	print(f"The maximum token length is {np.max(list(map(len, train_tokens)))}")
	train_tokens = list(map(lambda t: [1] + t[:max_length - 2] + [2], train_tokens))
	test_tokens = list(map(lambda t: [1] + t[:max_length - 2] + [2], test_tokens))

	train_tokens_ids = pad_sequences(train_tokens, maxlen = max_length, truncating="post", padding="post", dtype="int")
	test_tokens_ids = pad_sequences(test_tokens, maxlen = max_length, truncating="post", padding="post", dtype="int")




	train_tokens_tensor = torch.tensor(train_tokens_ids)
	train_y_tensor = torch.tensor(np.array(label_train)).long()

	test_tokens_tensor = torch.tensor(test_tokens_ids)
	test_y_tensor = torch.tensor(np.array(label_test)).long()


	train_dataset = TensorDataset(train_tokens_tensor, train_y_tensor)
	train_sampler = RandomSampler(train_dataset)
	train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, pin_memory=True)

	PART_OF_TEST = 50
	idxs = random.sample(range(len(test_tokens_tensor)), len(test_tokens_tensor)//PART_OF_TEST)
	val_dataset = TensorDataset(test_tokens_tensor[idxs], test_y_tensor[idxs])
	val_sampler = SequentialSampler(val_dataset)
	val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size, pin_memory=True)

	test_dataset = TensorDataset(test_tokens_tensor, test_y_tensor)
	test_sampler = SequentialSampler(test_dataset)
	test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size, pin_memory=True)


	return train_dataloader, val_dataloader, test_dataloader, LABELS, label_test
