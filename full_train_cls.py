#COLORS
from colors import ATTRIBUTES, FOREGROUND_COLORS, RESET

#STD
import os
import torch


#CUSTOM
from lib.configs import DataConfig, ModelConfig
from lib.utils import model_size, print_model_params, trainer, evaluator, calculate_metrics, metrics_to_str
from lib.preprocessing import dataset_selector, make_tokenizer, encode_dataset
from lib.Models.models import Brav, Bert_efficient, Nano_Bert_Efficient
from lib.Models.final_classifiers import Classifier_rms, Classifier_BERT, Classifier_post_electra




########################################################################################
EPOCHS = 1
LR = 1e-2

dataset_config = DataConfig(
                    dataset_name=None, 
                    dict_size=pow(2, 12), 
                    tokenizer_type="bpe", 
                    batch_size=128, 
                    max_len=512, 
                    labels=None
                )


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
########################################################################################


for DATASET_NAME in ["dbpedia","imdb", "sst2", "news", "bull", "limit",  "nlu", "snips", "multi_nli"]:
	dataset_config.dataset_name = DATASET_NAME


	print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS["BrightYellow"]}Loading dataset {DATASET_NAME} {RESET}")
	
	print(f"{FOREGROUND_COLORS["BrightCyan"]}", end="")
	train_dataset, test_dataset = dataset_selector(dataset_config.dataset_name)
	dataset_config.labels = train_dataset.unique("label")
	
	tokenizer = make_tokenizer(dataset_config, train_dataset)

	validation_dataset = train_dataset.train_test_split(test_size=0.1)
	train_dataset, validation_dataset = validation_dataset["train"], validation_dataset["test"]
	print(train_dataset)
	#limit train_dataset to 1000 samples
	train_dataset = train_dataset.select(list(range(1000)))

	train_dataloader = encode_dataset(tokenizer, train_dataset, dataset_config.max_len, dataset_config.batch_size)
	validation_dataloader = encode_dataset(tokenizer, validation_dataset, dataset_config.max_len, dataset_config.batch_size)
	test_dataloader = encode_dataset(tokenizer, test_dataset, dataset_config.max_len, dataset_config.batch_size)

	train_dataloader.shuffle = True

	models = [
		Brav,
		Bert_efficient,
		Nano_Bert_Efficient
	]

	configs = [
		ModelConfig( model_name="Brav", embedding_dimension=32, reduced_embedding_dimension=16, number_of_heads=None,  
		   			forward_expansion=8, num_layers=1, max_length=dataset_config.max_len, vocab_size=dataset_config.dict_size),
		ModelConfig( model_name="Bert_efficient", embedding_dimension=128, reduced_embedding_dimension=16, number_of_heads=None,
		   			forward_expansion=2, num_layers=1, max_length=dataset_config.max_len, vocab_size=dataset_config.dict_size),
		ModelConfig( model_name="Nano_Bert_Efficient", embedding_dimension=128, reduced_embedding_dimension=16, number_of_heads=None,
		   			forward_expansion=2, num_layers=1, max_length=dataset_config.max_len, vocab_size=dataset_config.dict_size)
	]
	
	for config in configs:
		if not os.path.exists(f"results/{config.model_name}/"):
				os.makedirs(f"results/{config.model_name}/")

		with open(f"results/{config.model_name}/{dataset_config.dataset_name}_{dataset_config.dict_size}_{dataset_config.tokenizer_type}_cls_report.txt", "a") as f:
			f.write(f"Model config for current run:\n{config}\n")
			f.write(f"LR: {LR}\n")
			f.write(f"EPOCHS: {EPOCHS}\n")
			f.write(f"\n*******************************************\n\n")


	for train_set in range(5):
		print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS["BrightYellow"]}------------------- Training session {train_set} -------------------{RESET}")
	

		for model_class, config in zip(models, configs):
			model = model_class(config)
			cls = Classifier_rms(model, config.embedding_dimension , dataset_config.n_labels())
			cls.to(device)
			
			########################################################################################
			print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS["BrightCyan"]}Model {model.__class__.__name__} parameters{RESET}")

			print(f"{FOREGROUND_COLORS["White"]}")
			print_model_params(model)
			print(f"{RESET}")

			########################################################################################
			print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS["BrightCyan"]}Training{RESET}")
			trainer(cls, train_dataloader, validation_dataloader, lr=LR, epochs=EPOCHS, logs_x_epoch=2)


			########################################################################################
			print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS["BrightCyan"]}Evaluation{RESET}")
			predicted, avg_eval_loss = evaluator(cls, test_dataloader)

			########################################################################################
			metrics = calculate_metrics(test_dataloader, predicted)
			print(metrics_to_str(metrics))

			########################################################################################
			
			#save the classification report in a file for later use specifying the dataset, model hyperparameters
			
			
	
			with open(f"results/{config.model_name}/{dataset_config.dataset_name}_{dataset_config.dict_size}_{dataset_config.tokenizer_type}_cls_report.txt", "a") as f:
				f.write(f"average eval loss: {avg_eval_loss: .4f}\n")
				f.write(f"{metrics_to_str(metrics)}\n")
				f.write(str(model_size(cls)))
				f.write("\n\n*******************************************\n\n")

	with open(f"results/{config.model_name}/{dataset_config.dataset_name}_{dataset_config.dict_size}_{dataset_config.tokenizer_type}_cls_report.txt", "a") as f:
		f.write(f"**************************************************\n==================================================\n**************************************************\n")
