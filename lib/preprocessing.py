from datasets import load_dataset, Dataset
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

	elif name == "sst2": #https://huggingface.co/datasets/stanfordnlp/sst2
		dataset = load_dataset("stanfordnlp/sst2", cache_dir="./datasets")

		train_data = dataset['train']
		test_data = dataset['validation']

		train_data = train_data.rename_column("sentence", "text").remove_columns("idx")
		test_data = test_data.rename_column("sentence", "text").remove_columns("idx")	
		 
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

	elif name == "dbpedia":	#https://huggingface.co/datasets/fancyzhx/dbpedia_14
		dataset = load_dataset("fancyzhx/dbpedia_14", cache_dir="./datasets")

		train_data = dataset['train']
		test_data = dataset['test']

		train_data = train_data.rename_column("content", "text").remove_columns("title")
		test_data = test_data.rename_column("content", "text").remove_columns("title")

		
	elif name == "nli":#https://huggingface.co/datasets/nyu-mll/multi_nli
		#volendo si può fare upgrade a https://huggingface.co/datasets/sentence-transformers/all-nli
		dataset = load_dataset("nyu-mll/multi_nli", cache_dir="./datasets")

		train_data = dataset['train']
		test_data = dataset['validation_matched']

		#TODO join premise and hypotesis in the same text with a separator
		train_data = train_data.map(lambda x: {'text': x['premise'] + " [SEP] " + x['hypothesis']})
		test_data = test_data.map(lambda x: {'text': x['premise'] + " [SEP] " + x['hypothesis']})

		train_data = train_data.remove_columns(['promptID', 'pairID', 'premise', 'premise_binary_parse', 'premise_parse', 'hypothesis', 'hypothesis_binary_parse', 'hypothesis_parse', 'genre'])
		test_data = test_data.remove_columns(['promptID', 'pairID', 'premise', 'premise_binary_parse', 'premise_parse', 'hypothesis', 'hypothesis_binary_parse', 'hypothesis_parse', 'genre'])


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

	else:
		raise ValueError("Dataset not found")

	return train_data, test_data


#function to create or load a tokenizer based on the type, dataset and dictionary size
def make_tokenizer(config: DataConfig, train_dataset:Dataset):
	try:
		tokenizer = Tokenizer.from_file(f"tokenizers/{config.tokenizer_type}_{config.dataset_name}_{config.dict_size}.json")
		
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
			trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[MASK]", "[SEP]"], vocab_size=config.dict_size,  limit_alphabet=config.dict_size)
		elif config.tokenizer_type == "wordpiece":
			trainer = WordPieceTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[MASK]", "[SEP]"], vocab_size=config.dict_size, limit_alphabet=config.dict_size)
		elif config.tokenizer_type == "unigram":
			trainer = UnigramTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[MASK]", "[SEP]"], vocab_size=config.dict_size)

		#train the tokenizer
		tokenizer.train_from_iterator(train_dataset['text'], trainer=trainer)

		assert(tokenizer.get_vocab_size() == config.dict_size)

		#save the tokenizer
		if not os.path.exists("tokenizers"):
			os.makedirs("tokenizers")
		tokenizer.save(f"tokenizers/{config.tokenizer_type}_{config.dataset_name}_{config.dict_size}.json")


	print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")
	assert(tokenizer.get_vocab_size() == config.dict_size)

	aux = tokenizer.encode_batch(train_dataset['text'][:100000])
	print(f"Average input length: {sum(map((lambda x: len(x.ids)), aux))/len(train_dataset['text'][:100000])}")
	print(f"Max input length: {max(map((lambda x: len(x.ids)), aux))}")
	print(f"Min input length: {min(map((lambda x: len(x.ids)), aux))}")
	del aux
	return tokenizer

	
# function to encode the text using the tokenizer and return a torch usable dataloader
def encode_dataset(tokenizer:Tokenizer, dataset:Dataset, max_length:int, batch_size:int):

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
		
	dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, pin_memory=True)

	dataloader.shuffle = False
	
	return dataloader