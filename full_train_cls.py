#COLORS
from colors import ATTRIBUTES, FOREGROUND_COLORS, RESET

#STD
import os
import torch


#CUSTOM
from lib.configs import DataConfig, ModelConfig
from lib.utils import model_size, print_model_params, trainer, evaluator, calculate_metrics, metrics_to_str
from lib.preprocessing import dataset_selector, make_tokenizer, encode_dataset
from lib.Models.final_classifiers import *
from lib.Models.models import *

from lib.trainer import Trainer


########################################################################################
EPOCHS = 20
LR = 5e-3

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


for DATASET_NAME in ["imdb", "sst2", "news", "bull", "limit", "nlu", "snips", "nli", "emotion_split"]:
	dataset_config.dataset_name = DATASET_NAME


	print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS["BrightYellow"]}Loading dataset {DATASET_NAME} {RESET}")
	
	print(f"{FOREGROUND_COLORS["BrightCyan"]}", end="")
	train_dataset, test_dataset = dataset_selector(dataset_config.dataset_name)
	dataset_config.labels = train_dataset.unique("label")
	
	tokenizer = make_tokenizer(dataset_config, train_dataset)

	validation_dataset = train_dataset.train_test_split(test_size=0.05)
	train_dataset, validation_dataset = validation_dataset["train"], validation_dataset["test"]


	train_dataloader = encode_dataset(tokenizer, train_dataset, dataset_config.max_len, dataset_config.batch_size)
	validation_dataloader = encode_dataset(tokenizer, validation_dataset, dataset_config.max_len, dataset_config.batch_size)
	test_dataloader = encode_dataset(tokenizer, test_dataset, dataset_config.max_len, dataset_config.batch_size)

	train_dataloader.shuffle = True

	models = [
		# Mlp_structured,
		# Nano_Mlp_structured,
		# Brav,
		# Bert_efficient,
		# Nano_Bert_Efficient,
		# Gray_BERT_Efficient,
		# Mamba_model,
		# MamBra_model,
		Embedder_model
	]

	configs = [
		# ModelConfig( model_name="Mlp_structured", embedding_dimension=64, reduced_embedding_dimension=16, number_of_heads=None,
		#    			forward_expansion=8, num_layers=1, max_length=dataset_config.max_len, vocab_size=dataset_config.dict_size),
		# ModelConfig( model_name="Nano_Mlp_structured", embedding_dimension=64, reduced_embedding_dimension=16, number_of_heads=None,
		#    			forward_expansion=4, num_layers=3, max_length=dataset_config.max_len, vocab_size=dataset_config.dict_size),
		# ModelConfig( model_name="Brav", embedding_dimension=32, reduced_embedding_dimension=16, number_of_heads=None,  
		#    			forward_expansion=8, num_layers=6, max_length=dataset_config.max_len, vocab_size=dataset_config.dict_size),
		# ModelConfig( model_name="Bert_efficient", embedding_dimension=128, reduced_embedding_dimension=16, number_of_heads=None,
		#    			forward_expansion=2, num_layers=1, max_length=dataset_config.max_len, vocab_size=dataset_config.dict_size),
		# ModelConfig( model_name="Nano_Bert_Efficient", embedding_dimension=64, reduced_embedding_dimension=16, number_of_heads=None,
		#    			forward_expansion=2, num_layers=2, max_length=dataset_config.max_len, vocab_size=dataset_config.dict_size),
		# ModelConfig( model_name="Gray_BERT_Efficient", embedding_dimension=64, reduced_embedding_dimension=16, number_of_heads=None,
		# 				forward_expansion=2, num_layers=4, max_length=dataset_config.max_len, vocab_size=dataset_config.dict_size)
		# ModelConfig( model_name="Mamba", embedding_dimension=64, reduced_embedding_dimension=16, number_of_heads=None,
		#    			 	forward_expansion=3, num_layers=2, max_length=dataset_config.max_len, vocab_size=dataset_config.dict_size),	
		# ModelConfig( model_name="MamBra", embedding_dimension=128, reduced_embedding_dimension=16, number_of_heads=None,
		#    			 	forward_expansion=0.25, num_layers=5, max_length=dataset_config.max_len, vocab_size=dataset_config.dict_size),	
		ModelConfig( model_name="Nano_Bert_Efficient_+_conv", embedding_dimension=64, reduced_embedding_dimension=16, number_of_heads=None,
		   			forward_expansion=2, num_layers=2, max_length=dataset_config.max_len, vocab_size=dataset_config.dict_size),
		ModelConfig( model_name="Mamba_+_conv", embedding_dimension=64, reduced_embedding_dimension=16, number_of_heads=None,
		   			 	forward_expansion=3, num_layers=2, max_length=dataset_config.max_len, vocab_size=dataset_config.dict_size),	
		# ModelConfig( model_name="Embedder_only", embedding_dimension=128, reduced_embedding_dimension=16, number_of_heads=None,
		#    			 	forward_expansion=1, num_layers=1, max_length=dataset_config.max_len, vocab_size=dataset_config.dict_size),	
		
			  	
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
			# cls = Conv_classifier(model, model_out_sz=config.embedding_dimension, labels_num=dataset_config.n_labels())
			cls.to(device)
			
			########################################################################################
			print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS["BrightCyan"]}Model {model.__class__.__name__} parameters{RESET}")

			print(f"{FOREGROUND_COLORS["White"]}")
			print_model_params(cls)
			print(f"{RESET}")

			########################################################################################
			print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS["BrightCyan"]}Training{RESET}")
			trainer = Trainer(cls, device, LR, config)
			trainer.train(train_dataloader, validation_dataloader, EPOCHS, 3)

			########################################################################################
			print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS["BrightCyan"]}Evaluation{RESET}")
			predicted, avg_eval_loss = trainer.evaluate(test_dataloader)

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
