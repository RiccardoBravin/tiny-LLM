from datasets import load_dataset, ClassLabel

def numericalize(label:str):
	#convert text to numerical values
	

	return 

#selects and loads the dataset based on the name
def dataset_selector(name:str):
	if name == "imdb":
		dataset = load_dataset("imdb", cache_dir="./datasets")
		print(dataset)

		train_data = dataset['train']
		test_data = dataset['test']

	elif name == "snips":
		dataset = load_dataset("benayas/snips", cache_dir="./datasets")

		#print(dataset.features)
		train_data = dataset['train']
		test_data = dataset['test']

		# making a mapping of labels to integers
		unique_labels = train_data.unique("category")
		label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
		print(f"Label to ID mapping: {label_to_id}")
		
		# mapping the labels to integers
		train_data = train_data.map(lambda sample: {'label': label_to_id[sample['category']]}, batched=False)
		test_data  = test_data.map( lambda sample: {'label': label_to_id[sample['category']]}, batched=False)

		train_data.remove_columns("category")
		test_data.remove_columns("category")

		# train_data.rename_column("category", "label")
		# test_data.rename_column("category", "label")
		

	return train_data, test_data


		
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))


from tokenizers.pre_tokenizers import Whitespace
tokenizer.pre_tokenizer = Whitespace()


from tokenizers.trainers import WordPieceTrainer
trainer = WordPieceTrainer(special_tokens=["[UNK]", "[CLS]", "[PAD]", "[MASK]"], vocab_size=2^12)


train_dataset, test_dataset = dataset_selector("snips")


tokenizer.train_from_iterator(train_dataset['text'], trainer=trainer)