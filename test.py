import torch
from lib.utils import *
from lib.Models.models import *

model_config = ModelConfig(
					model_name=None,
					embedding_dimension=128,
					reduced_embedding_dimension=16,
					number_of_heads=1,
					max_length=256,
					forward_expansion=0.7,
					d_state=None,
					num_layers=4,
					vocab_size=pow(2, 13),
					learning_rate = 2e-4,
				)

model = Nano_Bert_Efficient(model_config, dropout=0.1)

checkpoint = f"{model.__class__.__name__}_24"   #CHANGE HERE THE CHECKPOINT TO USE <-----------------------
print(f"Loading checkpoint {checkpoint}")
resume(model, f"trained_models/checkpoints/{checkpoint}.pth")


print(model.embedder.token.weight)

#calculate average magnitute of the weights
print(torch.mean(torch.abs(model.embedder.token.weight)))