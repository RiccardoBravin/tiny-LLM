from colors import ATTRIBUTES, FOREGROUND_COLORS, RESET

from datasets import load_dataset, Dataset, concatenate_datasets
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
import os, random

from lib.configs import DataConfig

#selects and loads the dataset based on the name. Outputs the train and test datasets with "text" and "label" columns
def dataset_selector(name:str):
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

	elif name == "bookcorpus":
		dataset = load_dataset("bookcorpus/bookcorpus", cache_dir="./datasets", trust_remote_code=True)

		train_data = dataset['train']#.select(range(100000))
		test_data = None

	else:
		raise ValueError("Dataset not found")

	#make dataset use only ascii characters
	# train_data = train_data.map(lambda x: {'text': x['text'].encode('ascii', 'ignore').decode()})
	# test_data = test_data.map(lambda x: {'text': x['text'].encode('ascii', 'ignore').decode()})

	return train_data, test_data


#function to create or load a tokenizer based on the type, dataset and dictionary size
def make_tokenizer(config: DataConfig, train_dataset:Dataset):
	try:
		tokenizer = Tokenizer.from_file(f"tokenizers/{config.tokenizer_type}_{config.dataset_name}_{config.dict_size}.json")
		assert(tokenizer.get_vocab_size() == config.dict_size)
		print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")
	except:
		from tokenizers.models import BPE, WordPiece, Unigram

		#select the tokenizer model
		if config.tokenizer_type == "bpe":
			tokenizer_model = BPE(unk_token="[UNK]")
		elif config.tokenizer_type == "wordpiece":
			tokenizer_model = WordPiece(unk_token="[UNK]")
		elif config.tokenizer_type == "unigram":
			tokenizer_model = Unigram()

		#initialize the tokenizer
		tokenizer = Tokenizer(tokenizer_model)

		#set the pre-tokenizer
		from tokenizers.pre_tokenizers import Whitespace
		tokenizer.pre_tokenizer = Whitespace()

		#set the normalizer to ensure the text is clean
		from tokenizers.normalizers import BertNormalizer
		tokenizer.normalizer = BertNormalizer(clean_text=True, handle_chinese_chars=True, strip_accents=True, lowercase=True)


		#select the trainer corresponding to the tokenizer model
		from tokenizers.trainers import WordPieceTrainer, BpeTrainer, UnigramTrainer
		if config.tokenizer_type == "bpe":
			trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[MASK]", "[SEP]"], vocab_size=config.dict_size)
		elif config.tokenizer_type == "wordpiece":
			trainer = WordPieceTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[MASK]", "[SEP]"], vocab_size=config.dict_size)
		elif config.tokenizer_type == "unigram":
			trainer = UnigramTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[MASK]", "[SEP]"], vocab_size=config.dict_size)

		#train the tokenizer
		tokenizer.train_from_iterator(train_dataset['text'], trainer=trainer)
		print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")

		try:
			assert(tokenizer.get_vocab_size() == config.dict_size)
		except:
			print(f"{FOREGROUND_COLORS["BrightRed"]}DICTIONARY SIZE TOO BIG FOR DATASET, RESORTING TO LOWER SIZE{RESET}")
			config.dict_size = tokenizer.get_vocab_size()

		#save the tokenizer
		if not os.path.exists("tokenizers"):
			os.makedirs("tokenizers")
		tokenizer.save(f"tokenizers/{config.tokenizer_type}_{config.dataset_name}_{config.dict_size}.json")


	# aux = tokenizer.encode_batch(train_dataset['text'][:100000])
	# print(f"Average input length: {sum(map((lambda x: len(x.ids)), aux))/len(train_dataset['text'][:100000])}")
	# print(f"Max input length: {max(map((lambda x: len(x.ids)), aux))}")
	# print(f"Min input length: {min(map((lambda x: len(x.ids)), aux))}")
	# del aux
	return tokenizer

def load_tokenizer(config: DataConfig, tokenizer_filename:str):
	tokenizer = Tokenizer.from_file(f"tokenizers/{tokenizer_filename}.json")
	assert(tokenizer.get_vocab_size() == config.dict_size)
	print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")
	
	return tokenizer

# function to encode the text using the tokenizer and return a torch usable dataloader
def encode_dataset(tokenizer:Tokenizer, dataset:Dataset, max_length:int, batch_size:int, shuffle:bool):
	

	from tokenizers.processors import TemplateProcessing
	tokenizer.post_processor = TemplateProcessing(
		single="[CLS] $A",
		special_tokens=[
			("[CLS]", tokenizer.token_to_id("[CLS]")),
		],
	)

	tokenizer.enable_padding(length=max_length, pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]")
	tokenizer.enable_truncation(max_length=max_length)


	def tokenize_function(examples):
		# Encode the texts
		encodings = tokenizer.encode_batch(examples["text"])
	
		# Create a dictionary to hold the tokenized data
		tokenized_data = {
			"tokens": [encoding.ids for encoding in encodings],
			"attention_mask": [encoding.attention_mask for encoding in encodings],
		}
		return tokenized_data


	tokenized_dataset = dataset.map(tokenize_function, batched=True)
	tokenized_dataset.set_format(type='torch', columns=['tokens', 'attention_mask', 'label'])

	dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, pin_memory=True, shuffle=shuffle)

	return dataloader


def encode_pretr_dataset(tokenizer:Tokenizer, dataset:Dataset, max_length:int, batch_size:int):
	


	from tokenizers.processors import TemplateProcessing
	tokenizer.post_processor = TemplateProcessing(
		single="[CLS] $A",
		pair="[CLS] $A [SEP] $B:1 [SEP]:1",
		special_tokens=[
			("[CLS]", tokenizer.token_to_id("[CLS]")),
			("[SEP]", tokenizer.token_to_id("[SEP]")),
		],
	)

	tokenizer.enable_padding(length=max_length, pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]")
	tokenizer.enable_truncation(max_length=max_length)

	
	def tokenize_function(examples):
		pairs = []
		labels = []
		texts = examples["text"]
		for i in range(len(texts) - 1):
			if random.random() < 0.5:
				# Randomly pair sentences
				pairs.append((texts[i], texts[random.randint(0, len(texts) - 1)]))
				labels.append(0)
			else:
				# Consecutive pair sentences
				pairs.append((texts[i], texts[i + 1]))
				labels.append(1)

		pairs.append((texts[-1], texts[0]))
		labels.append(0)

		# Encode the pairs of texts
		encodings = tokenizer.encode_batch(pairs)

		# Create a dictionary to hold the tokenized data
		tokenized_data = {
			"tokens": [encoding.ids for encoding in encodings],
			"attention_mask": [encoding.attention_mask for encoding in encodings],
			"type_ids": [encoding.type_ids for encoding in encodings],
			"special_tokens_mask": [encoding.special_tokens_mask for encoding in encodings],
			"label": labels,
		}

		return tokenized_data
	


	tokenized_dataset = dataset.map(tokenize_function, batched=True, batch_size=50000, num_proc=8)

	tokenized_dataset.set_format(type='torch', columns=['tokens', 'attention_mask', 'special_tokens_mask', 'label'])
	
	# print(tokenized_dataset[0:32]["label"])
	# print(tokenizer.decode_batch(tokenized_dataset[0:32]['tokens'].tolist()))

	dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)

	dataloader.shuffle = True

	return dataloader