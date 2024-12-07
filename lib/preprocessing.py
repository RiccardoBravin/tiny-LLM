from lib.colors import ATTRIBUTES, FOREGROUND_COLORS, RESET

from datasets import load_dataset, Dataset, concatenate_datasets
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing

from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast
from typing import List, Union, Any, Dict

import torch

import os, random


#selects and loads the dataset based on the name. Outputs the train and test datasets with "text" and "label" columns
def dataset_selector(name:str, reduced:bool=False):
	#"imdb", "sst2", "news","bull", "limit", "dbpedia", "nli", "nlu" "snips"
	if name == "imdb": #https://huggingface.co/datasets/stanfordnlp/imdb
		dataset = load_dataset("stanfordnlp/imdb", cache_dir="./datasets")

		train_data = dataset['train']
		test_data = dataset['test']

	elif name == "news":#https://huggingface.co/datasets/fancyzhx/ag_news
		dataset = load_dataset("fancyzhx/ag_news", cache_dir="./datasets")

		train_data = dataset['train']
		test_data = dataset['test']

	elif name == "bull":
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

		label_mapping = {label: i for i, label in enumerate(set(label))}
		print(label_mapping)

		# Transform the labels to integers
		label = [label_mapping[l] for l in label]

		combined = list(zip(text, label))
		random.shuffle(combined)
		text_train, label_train = zip(*combined)


		dataset = {
			"text": text_train,
			"label": label_train
		}


		dataset = Dataset.from_dict(dataset)
		dataset = dataset.train_test_split(test_size=0.1)

		train_data = dataset['train']
		test_data = dataset['test']

	elif name == "limit": #https://huggingface.co/datasets/IBM/limit
		dataset = load_dataset("IBM/limit", cache_dir="./datasets", trust_remote_code=True)

		train_data = dataset['train']
		test_data = dataset['test']

		train_data = train_data.rename_column("sentence", "text")
		test_data = test_data.rename_column("sentence", "text")

		#map each label to an integer
		unique_labels = train_data.unique("motion")
		label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
		print(f"Label to ID mapping: {label_to_id}")

		# mapping the labels to integers
		train_data = train_data.map(lambda sample: {'label': label_to_id[sample['motion']]}, batched=False)
		test_data  = test_data.map( lambda sample: {'label': label_to_id[sample['motion']]}, batched=False)

		train_data = train_data.remove_columns(["id", "motion", "motion_entities"])
		test_data = test_data.remove_columns(["id", "motion", "motion_entities"])

	elif name == "nlu": #https://huggingface.co/datasets/xingkunliuxtracta/nlu_evaluation_data
		dataset = load_dataset("xingkunliuxtracta/nlu_evaluation_data", cache_dir="./datasets", trust_remote_code=True)

		train_data = dataset['train']

		unique_labels = train_data.unique("scenario")
		label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
		print(f"Label to ID mapping: {label_to_id}")

		# mapping the labels to integers
		train_data = train_data.map(lambda sample: {'label': label_to_id[sample['scenario']]}, batched=False)

		# cleanup useless columns
		train_data = train_data.remove_columns("scenario")

		#split the data into train and test
		dataset = train_data.train_test_split(test_size=0.1)

		train_data = dataset['train']
		test_data = dataset['test']

	elif name == "snips": # https://huggingface.co/datasets/benayas/snips
		dataset = load_dataset("benayas/snips", cache_dir="./datasets")

		train_data = dataset['train']
		test_data = dataset['test']

		# making a mapping of labels to integers
		unique_labels = train_data.unique("category")
		label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
		print(f"Label to ID mapping: {label_to_id}")

		# mapping the labels to integers
		train_data = train_data.map(lambda sample: {'label': label_to_id[sample['category']]}, batched=False)
		test_data  = test_data.map( lambda sample: {'label': label_to_id[sample['category']]}, batched=False)

		# cleanup useless columns
		train_data = train_data.remove_columns("category")
		test_data = test_data.remove_columns("category")

	elif name == "qnli":
		dataset = load_dataset("glue", "qnli", cache_dir="./datasets")

		train_data = dataset['train']
		test_data = dataset['validation']

		#join premise and hypotesis in the same text with a separator
		train_data = train_data.map(lambda x: {'text': x['question'] + " [SEP] " + x['sentence']})
		test_data = test_data.map(lambda x: {'text': x['question'] + " [SEP] " + x['sentence']})

		train_data = train_data.remove_columns(['question', 'sentence', 'idx'])
		test_data = test_data.remove_columns(['question', 'sentence', 'idx'])


	elif name == "emotion_split": #https://huggingface.co/datasets/dair-ai/emotion
		dataset = load_dataset("dair-ai/emotion", "split", cache_dir="./datasets")

		train_data = concatenate_datasets([dataset['train'], dataset['validation']])
		test_data = dataset['test']

	elif name == "emotion_unsplit": #https://huggingface.co/datasets/dair-ai/emotion
		dataset = load_dataset("dair-ai/emotion", "unsplit", cache_dir="./datasets")
		dataset = dataset['train']

		dataset = dataset.train_test_split(test_size=0.1)

		train_data = dataset['train']
		test_data = dataset['test']



	#STARTIN GLUE DATASET



	elif name == "cola": # https://huggingface.co/datasets/nyu-mll/glue/viewer/cola
		dataset = load_dataset("nyu-mll/glue", "cola", cache_dir="./datasets")

		train_data = dataset['train']
		test_data = dataset['validation']

		train_data = train_data.rename_column("sentence", "text").remove_columns("idx")
		test_data = test_data.rename_column("sentence", "text").remove_columns("idx")


	elif name == "mnli-m":#https://huggingface.co/datasets/nyu-mll/multi_nli
		#volendo si può fare upgrade a https://huggingface.co/datasets/sentence-transformers/all-nli
		dataset = load_dataset("nyu-mll/glue", "mnli", cache_dir="./datasets")

		train_data = dataset['train']
		test_data = dataset['validation_matched']

		#join premise and hypotesis in the same text with a separator
		train_data = train_data.map(lambda x: {'text': x['premise'] + " [SEP] " + x['hypothesis']})
		test_data = test_data.map(lambda x: {'text': x['premise'] + " [SEP] " + x['hypothesis']})

		train_data = train_data.remove_columns(['premise', 'hypothesis', 'idx'])
		test_data = test_data.remove_columns(['premise', 'hypothesis', 'idx'])

	elif name == "mnli-mm":#https://huggingface.co/datasets/nyu-mll/multi_nli
		dataset = load_dataset("nyu-mll/glue", "mnli", cache_dir="./datasets")

		train_data = dataset['train']
		test_data = dataset['validation_mismatched']

		#join premise and hypotesis in the same text with a separator
		train_data = train_data.map(lambda x: {'text': x['premise'] + " [SEP] " + x['hypothesis']})
		test_data = test_data.map(lambda x: {'text': x['premise'] + " [SEP] " + x['hypothesis']})

		train_data = train_data.remove_columns(['premise', 'hypothesis', 'idx'])
		test_data = test_data.remove_columns(['premise', 'hypothesis', 'idx'])

	elif name == "mrpc":#https://huggingface.co/datasets/nyu-mll/multi_nli
		dataset = load_dataset("nyu-mll/glue", "mrpc", cache_dir="./datasets")

		train_data = dataset['train']
		test_data = dataset['validation']

		train_data = train_data.map(lambda x: {'text': x['sentence1'] + " [SEP] " + x['sentence2']})
		test_data = test_data.map(lambda x: {'text': x['sentence1'] + " [SEP] " + x['sentence2']})

		train_data = train_data.remove_columns(['sentence1', 'sentence2', 'idx'])
		test_data = test_data.remove_columns(['sentence1', 'sentence2', 'idx'])

	elif name == "qnli":#https://huggingface.co/datasets/nyu-mll/multi_nli
		dataset = load_dataset("nyu-mll/glue", "qnli", cache_dir="./datasets")

		train_data = dataset['train']
		test_data = dataset['validation']

		train_data = train_data.map(lambda x: {'text': x['question'] + " [SEP] " + x['sentence']})
		test_data = test_data.map(lambda x: {'text': x['question'] + " [SEP] " + x['sentence']})

		train_data = train_data.remove_columns(['question', 'sentence', 'idx'])
		test_data = test_data.remove_columns(['question', 'sentence', 'idx'])

	elif name == "qqp":#https://huggingface.co/datasets/nyu-mll/multi_nli
		dataset = load_dataset("nyu-mll/glue", "qqp", cache_dir="./datasets")

		train_data = dataset['train']
		test_data = dataset['validation']

		#join premise and hypotesis in the same text with a separator
		train_data = train_data.map(lambda x: {'text': x['question1'] + " [SEP] " + x['question2']})
		test_data = test_data.map(lambda x: {'text': x['question1'] + " [SEP] " + x['question2']})

		train_data = train_data.remove_columns(['question1', 'question2', 'idx'])
		test_data = test_data.remove_columns(['question1', 'question2', 'idx'])

	elif name == "rte":#https://huggingface.co/datasets/nyu-mll/multi_nli
		dataset = load_dataset("nyu-mll/glue", "rte", cache_dir="./datasets")

		train_data = dataset['train']
		test_data = dataset['validation']

		#join premise and hypotesis in the same text with a separator
		train_data = train_data.map(lambda x: {'text': x['sentence1'] + " [SEP] " + x['sentence2']})
		test_data = test_data.map(lambda x: {'text': x['sentence1'] + " [SEP] " + x['sentence2']})

		train_data = train_data.remove_columns(['sentence1', 'sentence2', 'idx'])
		test_data = test_data.remove_columns(['sentence1', 'sentence2', 'idx'])


	elif name == "sst2":
		dataset = load_dataset("nyu-mll/glue", "sst2", cache_dir="./datasets")

		train_data = dataset['train']
		test_data = dataset['validation']

		train_data = train_data.rename_column("sentence", "text").remove_columns("idx")
		test_data = test_data.rename_column("sentence", "text").remove_columns("idx")

	elif name == "stsb":#https://huggingface.co/datasets/nyu-mll/multi_nli  # 		WARNING THIS DATASET IS FOR REGRESSION NOT CLASIFICATION
		dataset = load_dataset("nyu-mll/glue", "stsb", cache_dir="./datasets")

		train_data = dataset['train']
		test_data = dataset['validation']

		#join premise and hypotesis in the same text with a separator
		train_data = train_data.map(lambda x: {'text': x['sentence1'] + " [SEP] " + x['sentence2']})
		test_data = test_data.map(lambda x: {'text': x['sentence1'] + " [SEP] " + x['sentence2']})

		train_data = train_data.remove_columns(['sentence1', 'sentence2', 'idx'])
		test_data = test_data.remove_columns(['sentence1', 'sentence2', 'idx'])

	elif name == "wnli":#https://huggingface.co/datasets/nyu-mll/multi_nli
		dataset = load_dataset("nyu-mll/glue", "wnli", cache_dir="./datasets")

		train_data = dataset['train']
		test_data = dataset['validation']

		#join premise and hypotesis in the same text with a separator
		train_data = train_data.map(lambda x: {'text': x['sentence1'] + " [SEP] " + x['sentence2']})
		test_data = test_data.map(lambda x: {'text': x['sentence1'] + " [SEP] " + x['sentence2']})

		train_data = train_data.remove_columns(['sentence1', 'sentence2', 'idx'])
		test_data = test_data.remove_columns(['sentence1', 'sentence2', 'idx'])

	else:
		raise ValueError("Dataset not found")

	#make dataset use only ascii characters
	# train_data = train_data.map(lambda x: {'text': x['text'].encode('ascii', 'ignore').decode()})
	# test_data = test_data.map(lambda x: {'text': x['text'].encode('ascii', 'ignore').decode()})

	return train_data, test_data


#function to create or load a tokenizer based on the type, dataset and dictionary size
def make_tokenizer(tokenizer_type:str, dictionary_size:int, dataset_name:str, train_dataset:Dataset):
	try:
		tokenizer = Tokenizer.from_file(f"tokenizers/{tokenizer_type}_{dataset_name}_{dictionary_size}.json")
			
		assert(tokenizer.get_vocab_size() == dictionary_size)
		print(f"Tokenizer loaded with vocab size: {tokenizer.get_vocab_size()}")
	except:
		from tokenizers.models import BPE, WordPiece, Unigram

		#select the tokenizer model
		if tokenizer_type == "bpe":
			tokenizer_model = BPE(unk_token="[UNK]")
		elif tokenizer_type == "wordpiece":
			tokenizer_model = WordPiece(unk_token="[UNK]")
		elif tokenizer_type == "unigram":
			tokenizer_model = Unigram()

		#initialize the tokenizer
		tokenizer = Tokenizer(tokenizer_model)

		#set the pre-tokenizer
		from tokenizers.pre_tokenizers import Whitespace
		tokenizer.pre_tokenizer = Whitespace()

		#set the normalizer to ensure the text is clean
		from tokenizers.normalizers import BertNormalizer
		tokenizer.normalizer = BertNormalizer(clean_text=True, handle_chinese_chars=True, strip_accents=True, lowercase=True)

		special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[MASK]", "[SEP]"]

		#select the trainer corresponding to the tokenizer model
		from tokenizers.trainers import WordPieceTrainer, BpeTrainer, UnigramTrainer
		if tokenizer_type == "bpe":
			trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=dictionary_size)
		elif tokenizer_type == "wordpiece":
			trainer = WordPieceTrainer(special_tokens=special_tokens, vocab_size=dictionary_size)
		elif tokenizer_type == "unigram":
			trainer = UnigramTrainer(special_tokens=special_tokens, vocab_size=dictionary_size)

		#train the tokenizer
		tokenizer.train_from_iterator(train_dataset['text'], trainer=trainer)
		print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")

		try:
			assert(tokenizer.get_vocab_size() == dict_size)
		except:
			print(f"{FOREGROUND_COLORS["BrightRed"]}DICTIONARY SIZE TOO BIG FOR DATASET, RESORTING TO LOWER SIZE{RESET}")
			dict_size = tokenizer.get_vocab_size()

		#save the tokenizer
		if not os.path.exists("tokenizers"):
			os.makedirs("tokenizers")
		tokenizer.save(f"tokenizers/{tokenizer_type}_{dataset_name}_{dict_size}.json")


	# aux = tokenizer.encode_batch(train_dataset['text'][:100000])
	# print(f"Average input length: {sum(map((lambda x: len(x.ids)), aux))/len(train_dataset['text'][:100000])}")
	# print(f"Max input length: {max(map((lambda x: len(x.ids)), aux))}")
	# print(f"Min input length: {min(map((lambda x: len(x.ids)), aux))}")
	# del aux


	tokenizer.post_processor = TemplateProcessing(
		single="[CLS] $A",
		special_tokens=[
			("[CLS]", tokenizer.token_to_id("[CLS]")),
		],
	)

	return PreTrainedTokenizerFast(
								tokenizer_object = tokenizer, 
								cls_token="[CLS]", 
								sep_token="[SEP]", 
								unk_token="[UNK]", 
								pad_token="[PAD]", 
								mask_token="[MASK]"
							)







def pretr_tokenizer(dataset:Dataset, dictionary_size:int, max_length:int):
	
	try:
		tokenizer = Tokenizer.from_file(f"tokenizers/bpe_book_corpus_{dictionary_size}.json")
			
		assert(tokenizer.get_vocab_size() == dictionary_size)
		print(f"\tFound tokenizer with {tokenizer.get_vocab_size()}")

	except:

		from tokenizers.models import BPE
		from tokenizers.trainers import BpeTrainer

		tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

		#set the pre-tokenizer
		from tokenizers.pre_tokenizers import Whitespace
		tokenizer.pre_tokenizer = Whitespace()

		#set the normalizer to ensure the text is clean
		from tokenizers.normalizers import BertNormalizer
		tokenizer.normalizer = BertNormalizer(clean_text=True, handle_chinese_chars=True, strip_accents=True, lowercase=True)


		special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[MASK]", "[SEP]"]
		trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=dictionary_size, show_progress=True)   

		tokenizer.train_from_iterator(dataset['text'], trainer=trainer)

		
		tokenizer.post_processor = TemplateProcessing(
			pair="[CLS] $A [SEP] $B:1 [SEP]:1",
			special_tokens=[
				("[CLS]", tokenizer.token_to_id("[CLS]")),
				("[SEP]", tokenizer.token_to_id("[SEP]")),
			],
		)

		if tokenizer.get_vocab_size() != dictionary_size:
			print(f"{FOREGROUND_COLORS['BrightRed']}DICTIONARY SIZE TOO BIG FOR DATASET, RESORTING TO LOWER SIZE {tokenizer.get_vocab_size()}{RESET}")

		#save the tokenizer
		if not os.path.exists("tokenizers"):
			os.makedirs("tokenizers")
		tokenizer.save(f"tokenizers/bpe_book_corpus_{tokenizer.get_vocab_size()}.json")


		
	return 	PreTrainedTokenizerFast(
					tokenizer_object = tokenizer, 
					cls_token="[CLS]", 
					sep_token="[SEP]", 
					unk_token="[UNK]", 
					pad_token="[PAD]", 
					mask_token="[MASK]",
					model_max_length=max_length
			)



def pretr_dataset_builder(dataset:Dataset):
	#modify the dataset to have a tet and a text_pair column for the next sentence prediction as well as a label_nsp column
	import numpy as np


	new_text = dataset['text'][:-1]
	new_text_pair = dataset['text'][1:]

	dataset_ordered = Dataset.from_dict({
		"text": new_text,
		"text_pair": new_text_pair,
		"label_nsp": np.zeros(len(new_text)).tolist()
	})
	del new_text
	print("\tOrdered dataset created")

	dataset = dataset.shuffle()


	new_text = dataset['text'][:-1]
	dataset_random = Dataset.from_dict({
		"text": new_text,
		"text_pair": new_text_pair,
		"label_nsp": np.ones(len(new_text)).tolist()
	})

	del new_text, new_text_pair, dataset
	print("\tRandomized dataset created")

	#join the two datasets
	train_data = concatenate_datasets([dataset_ordered, dataset_random]).shuffle()
	print("\tDatasets joined")

	# Splitting the dataset
	eval_data = train_data.train_test_split(test_size=1000)
	return eval_data["train"], eval_data["test"]



# Masked language modeling and NSP collator
class MlmNspCollator(DataCollatorForLanguageModeling):
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        text = [example["text"] for example in examples]
        text_pair = [example["text_pair"] for example in examples]
        nsp_labels = [example["label_nsp"] for example in examples]
        batch = self.tokenizer(text=text, text_pair=text_pair, padding=True, truncation=True, return_tensors="pt")
        batch.pop("token_type_ids")  # Not required for pretraining
        batch["label_nsp"] = torch.tensor(nsp_labels, dtype=torch.long) 


        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch
