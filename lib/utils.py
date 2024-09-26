from colors import ATTRIBUTES, FOREGROUND_COLORS, RESET

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix

from tqdm import tqdm
import torch
import torch.nn as nn 

from torch.utils.data import Dataset

# Print the model size
def model_size(model):
	param_size = 0
	param_count = 0
	for param in model.parameters():
		param_size += param.nelement() * param.element_size()
		param_count += param.nelement()
	buffer_size = 0 
	for buffer in model.buffers():
		buffer_size += buffer.nelement() * buffer.element_size()
		
	size_all_mb = (param_size + buffer_size) / 1024**2

	if param_count < 1024*1024:
		out_string = "Model params: {:.3f}K".format(param_count/1e3), "Model buffers: {:.3f}K".format(buffer_size/1e3), "Model size: {:.3f}MB".format(size_all_mb)
	else:
		out_string = "Model params: {:.3f}M".format(param_count/1e6), "Model buffers: {:.3f}K".format(buffer_size/1e3), "Model size: {:.3f}MB".format(size_all_mb)
	
	return out_string

def print_model_params(model):
	#model class name
	print(f"Model name: {model.__class__.__name__}")
	#all layer parameters
	for name, param in model.named_parameters():
		print(f"{name}: {param.nelement()}")
	#full model size
	print(model_size(model))

def activations_calculator(model, dict_size, max_len):
	# Register hooks
	activations = {}
	model_copy = model 
	
	def get_activation(name):
		def hook(model, input, output):
			activations[name] = output.shape
		return hook
	
	for name, layer in model_copy.named_modules():
		layer.register_forward_hook(get_activation(name))
	
	# Pass a dummy input through the model
	dummy_input = torch.randint(0, dict_size, (1, max_len)).to("cuda")  # Example input size (batch_size, max_length, dict_size)
	model_copy(dummy_input)
	
	# Print the captured activation sizes
	for name, shape in activations.items():
		print(f"Layer: {name}, Output shape: {shape}")


def n_ary_gray_code(n, base = 3):
	# n x n**3 list 
	gray = [[0] * n for _ in range(base**n)]
	for j in range(n):
		i = 0
		val = 0
		invert = True
		while i < base**n:
			for k in range(base**j):
				# print(i+k)
				gray[i+k][j] = val
			
			i += base**j
			
			
			if  invert:
				val += 1
			else:
				val -= 1
			
			if val == base:
				invert = not invert
				val = base - 1
			elif val == -1:
				invert = not invert
				val = 0

	return gray



def calculate_metrics(test_dataloader, predicted):
	labels = []
	for batch in test_dataloader:
		labels.extend(batch["label"].tolist())

	try:
		accuracy = accuracy_score(labels, predicted)
		f1 = f1_score(labels, predicted, average='weighted')  
		precision = precision_score(labels, predicted, average='weighted')  
		recall = recall_score(labels, predicted, average='weighted')
		mcc = matthews_corrcoef(labels, predicted)
		conf_mat = confusion_matrix(labels, predicted)
	
		metrics_dict = {
			"accuracy": accuracy,
			"f1": f1,
			"precision": precision,
			"recall": recall,
			"mcc": mcc,
			"conf_mat": conf_mat

		}
	except:
		spe_corr = spearman_correlation(torch.tensor(labels), torch.tensor(predicted))
		metrics_dict = {
			"spearman_correlation": spe_corr
		}
	return metrics_dict

def metrics_to_str(metrics):
	try:
		acc_str = f"Accuracy: {metrics['accuracy']:.3f}"
		f1_str = f"F1: {metrics['f1']:.3f}"
		prec_str = f"Precision: {metrics['precision']:.3f}"
		rec_str = f"Recall: {metrics['recall']:.3f}"
		mcc_str = f"MCC: {metrics['mcc']:.3f}"
		conf_mat_str = f"Confusion matrix:\n {metrics['conf_mat']}"

		out_str = f"{acc_str}\n{f1_str}\n{prec_str}\n{rec_str}\n{mcc_str}\n{conf_mat_str}"
	except:
		spe_corr_str = f"Spearman correlation coefficient: {metrics['spearman_correlation']:.3f}"
		out_str = f"{spe_corr_str}"
	return out_str 


def _get_ranks(x: torch.Tensor) -> torch.Tensor:
    tmp = x.argsort()
    ranks = torch.zeros_like(tmp)
    ranks[tmp] = torch.arange(len(x))
    return ranks

def spearman_correlation(x: torch.Tensor, y: torch.Tensor):
    """Compute correlation between 2 1-D vectors
    Args:
        x: Shape (N, )
        y: Shape (N, )
    """
    x_rank = _get_ranks(x)
    y_rank = _get_ranks(y)
    
    n = x.size(0)
    upper = 6 * torch.sum((x_rank - y_rank).pow(2))
    down = n * (n ** 2 - 1.0)
    return 1.0 - (upper / down)


def checkpoint(model, filename):
	torch.save(model.state_dict(), filename)

def resume(model, filename):
	model.load_state_dict(torch.load(filename))