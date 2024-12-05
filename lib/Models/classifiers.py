	
# class Classifier_first_token(nn.Module):

#     def __init__(self, model:nn.Module, model_out_sz: int, labels_num:int):
#         r"""
#         Classifier that takes only the last token of the sequence to classify the data
#         Args:
#             model: the model that will be used to generate the embeddings
#             model_out_sz: the output size of the model
#             labels_num: the number of labels to output
#         """
#         super().__init__()
#         self.model = model
#         self.act = nn.Tanh()
#         self.fc = nn.Linear(model_out_sz, labels_num)

#     def forward(self, x:torch.Tensor, mask:torch.Tensor):
#         x = self.model(x, mask)
#         x = x[:,0]
#         x = self.act(x)
#         x = self.fc(x)
#         return x    
	

# class Classifier_rms(nn.Module):

#     def __init__(self, model:nn.Module, model_out_sz: int, labels_num:int):
#         r"""
#         Classifier that uses the root mean square of the embeddings along the sequence length to classify the data
#         Args:
#             model: the model that will be used to generate the embeddings
#             model_out_sz: the output size of the model
#             labels_num: the number of labels to output
#         """
#         super().__init__()
#         self.model = model
#         self.act = nn.Sigmoid()
#         self.fc = nn.Linear(model_out_sz, labels_num)

#     def forward(self, x:torch.Tensor, mask:torch.Tensor):
#         x = self.model(x, mask)

#         x = torch.sqrt(torch.mean(torch.pow(x, 2), dim=1))

#         x = self.act(x)
#         x = self.fc(x)
#         return x
	

# class Classifier_BERT_pretraining(nn.Module):

#     def __init__(self, model:nn.Module, model_out_sz: int, dictionary_size:int, num_labels:int):
#         r"""
#         Classifier that takes only the firts token of the sequence to classify the data
#         Args:
#             model: the model that will be used to generate the embeddings
#             model_out_sz: the output size of the model
#             labels_num: the number of labels to output
#         """
#         super().__init__()
#         self.model = model

#         self.lm_cls = nn.Linear(model_out_sz, dictionary_size)


#         self.class_cls = nn.Sequential(
#             nn.Tanh(),
#             nn.Linear(model_out_sz, num_labels),
#         )

#         self.lm_cls.weight = self.model.embedder.token.weight

#     def forward(self, x:torch.Tensor, mask:torch.Tensor):
#         x = self.model(x, mask)


#         y1 = self.lm_cls(x)

#         x = x[:,0,:]
#         y2 = self.class_cls(x)

#         return y1, y2


# class Classifier_Nano_BERT_pretraining(nn.Module):

#     def __init__(self, model:nn.Module, model_out_sz: int, reduced_embedding_dimension: int,  dictionary_size:int, num_labels:int):
#         r"""
#         Classifier that takes only the firts token of the sequence to classify the data
#         Args:
#             model: the model that will be used to generate the embeddings
#             model_out_sz: the output size of the model
#             labels_num: the number of labels to output
#         """
#         super().__init__()
#         self.model = model

#         self.reducer = nn.Linear(model_out_sz, reduced_embedding_dimension, bias=False)
#         self.lm_cls = nn.Linear(reduced_embedding_dimension, dictionary_size)

#         self.class_cls = nn.Sequential(
#             nn.Tanh(),
#             nn.Linear(model_out_sz, num_labels),
#         )

#         # self.reducer.weight = torch.nn.Parameter(self.model.embedder.tok_expander.weight.T)
#         self.lm_cls.weight = self.model.embedder.token.weight

#     def forward(self, x:torch.Tensor, mask:torch.Tensor):
#         x = self.model(x, mask)

#         y1 = self.reducer(x)
#         y1 = self.lm_cls(y1)

#         x = x[:,0,:]
#         y2 = self.class_cls(x)

#         return y1, y2
	

from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput, MaskedLMOutput
from torch import nn
import torch

from lib.Models.BERT import BERT_Config, BERT
from lib.Models.mamba import MAMBA_Config, MAMBA
from lib.Models.NanoEmbedder import NanoEmbedder_Config, NanoEmbedder
from lib.Models.NanoEmbedderConv import NanoEmbedderConv_Config, NanoEmbedderConv
from lib.Models.NanoBERT import NanoBERT_Config, NanoBERT
from lib.Models.BERTEfficient import BERTEfficient_Config, BERTEfficient 
from lib.Models.NanoBERTEfficient import NanoBERTEfficient_Config, NanoBERTEfficient
from lib.Models.EmbBERT import EmbBERT_Config, EmbBERT

class SequenceClassifier(PreTrainedModel):
	def __init__(self, config):
		super().__init__(config)
		
		# Instantiate a model class based on the config model type
		model_class = globals()[config.model_type]
		self.model = model_class(config)
		
		self.classifier = nn.Linear(config.hidden_size, config.num_labels)
		self.celoss = nn.CrossEntropyLoss()
		self.mseloss = nn.MSELoss()
		


	def forward(self, input_ids, attention_mask, labels=None):
		outputs = self.model(input_ids, attention_mask)
		outputs = outputs[:,0]


		# Classification head
		logits = self.classifier(outputs)

		loss = None
		if labels is not None:
			try:
				loss = self.celoss(logits.view(-1, self.config.num_labels), labels.view(-1)) 
			except:
				loss = self.mseloss(logits.view(-1), labels.view(-1) )
			
		return SequenceClassifierOutput(loss=loss, logits=logits)
	
	#function to reinitialize the classifier layer weights
	def change_internal_model(self, model):
		self.model = model
	
class PretrainingClassifier(PreTrainedModel):
	def __init__(self, config):
		super().__init__(config)
		
		# Instantiate a model class based on the config model type
		model_class = globals()[config.model_type]
		self.model = model_class(config)
		
		self.nsp_classifier = nn.Linear(config.hidden_size, 2)
		self.mlm_classifier = nn.Linear(config.hidden_size, config.vocab_size)

		self.loss = nn.CrossEntropyLoss()

		


	def forward(self, input_ids, attention_mask, labels=None, label_nsp=None):
		outputs = self.model(input_ids, attention_mask)

		# Classification head
		logits_nsp = self.nsp_classifier(outputs[:,0])
		logits_mlm = self.mlm_classifier(outputs)

		loss = None
		if labels is not None and label_nsp is not None:
			mlm_loss = self.loss(logits_mlm.view(-1, self.config.vocab_size), labels.view(-1))
			nsp_loss = self.loss(logits_nsp.view(-1, 2), label_nsp.view(-1))
			loss = mlm_loss + nsp_loss

		return MaskedLMOutput(loss=loss, logits=logits_mlm)



class RMSClassifier(PreTrainedModel):
	def __init__(self, config):
		super().__init__(config)
		
		# Instantiate a model class based on the config model type
		model_class = globals()[config.model_type]
		self.model = model_class(config)
		
		self.classifier = nn.Linear(config.hidden_size, config.num_labels)
		self.celoss = nn.CrossEntropyLoss()
		self.mseloss = nn.MSELoss()
		


	def forward(self, input_ids, attention_mask, labels=None):
		outputs = self.model(input_ids, attention_mask)

		x = torch.sqrt(torch.mean(torch.pow(outputs, 2), dim=1))
		# Classification head
		logits = self.classifier(x)

		loss = None
		if labels is not None:
			try:
				loss = self.celoss(logits.view(-1, self.config.num_labels), labels.view(-1)) 
			except:
				loss = self.mseloss(logits.view(-1), labels.view(-1) )
			
		return SequenceClassifierOutput(loss=loss, logits=logits)
	
	#function to reinitialize the classifier layer weights
	def change_internal_model(self, model):
		self.model = model



# class Classifier_rms(nn.Module):

#     def __init__(self, model:nn.Module, model_out_sz: int, labels_num:int):
#         r"""
#         Classifier that uses the root mean square of the embeddings along the sequence length to classify the data
#         Args:
#             model: the model that will be used to generate the embeddings
#             model_out_sz: the output size of the model
#             labels_num: the number of labels to output
#         """
#         super().__init__()
#         self.model = model
#         self.act = nn.Sigmoid()
#         self.fc = nn.Linear(model_out_sz, labels_num)

#     def forward(self, x:torch.Tensor, mask:torch.Tensor):
#         x = self.model(x, mask)

#         x = torch.sqrt(torch.mean(torch.pow(x, 2), dim=1))

#         x = self.act(x)
#         x = self.fc(x)
#         return x