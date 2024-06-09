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
	return "Model params: {:.3f}M".format(param_count/1e6), "Model buffers: {:.3f}M".format(buffer_size/1e6), "Model size: {:.3f}MB".format(size_all_mb)


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
	dummy_input = torch.randn(1, max_len, dict_size)  # Example input size (batch_size, channels, height, width)
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



def trainer(model, train_dataloader, val_dataloader, lr, epochs, logs_x_epoch = 10):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	criterion = torch.nn.CrossEntropyLoss()
	criterion.to(device);

	optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
	#scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataloader), num_training_steps=epochs*len(train_dataloader))

	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=logs_x_epoch)
	#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*2, eta_min=1e-6)

	log_step = len(train_dataloader) // logs_x_epoch

	for epoch in range(epochs):
		tqdm.write(f"{FOREGROUND_COLORS['Green']}Epoch {epoch+1}/{epochs}")
		
		model.train()
		train_loss = 0
		tqdm_train_loader = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

		for step_num, batch_data in enumerate(tqdm_train_loader):

			tokens = batch_data["tokens"].to(device) 
			masks  = batch_data["attention_mask"].to(device)
			labels = batch_data["label"].to(device)
			
			logits = model(tokens, masks)

			batch_loss = criterion(logits, labels)
			if step_num == 0:
				train_loss = batch_loss.item()
			else:
				train_loss = 0.9 * train_loss + 0.1 * batch_loss.item()

			model.zero_grad()
			batch_loss.backward()


			#torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optimizer.step()
			#scheduler.step()

			loss_str = "{:.4f}".format(train_loss)
			tqdm_train_loader.set_postfix(loss = loss_str)			
			if step_num % log_step == (log_step - 1):
				model.eval()
				val_accuracy = []
				guesses = []
				val_loss = 0
				for batch_val in val_dataloader:
					tokens = batch_val["tokens"].to(device) 
					masks  = batch_val["attention_mask"].to(device)
					labels_val = batch_val["label"].to(device)

					guess = model(tokens, masks)
					val_loss += criterion(guess, labels_val).item()
					val_accuracy += (torch.argmax(guess, dim=1) == labels_val).tolist()
					guesses += torch.argmax(guess, dim=1).tolist()

				val_accuracy = sum(val_accuracy) / len(val_accuracy)
				mcc = matthews_corrcoef(val_dataloader.dataset["label"], guesses)
				val_loss = val_loss / len(val_dataloader)
				scheduler.step(-mcc)
				tqdm.write(f"{RESET}Val loss: {val_loss:.3f}, Val accuracy: {val_accuracy:.3f}, Val mcc: {mcc:.3f}, Lr: {scheduler.get_last_lr()} {FOREGROUND_COLORS['Green']}")
				model.train()
			
	print(f"{RESET}")		



def evaluator(model, test_dataloader):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	criterion = torch.nn.CrossEntropyLoss()
	criterion.to(device);

	model.eval()
	predicted = torch.Tensor([])
	all_logits = torch.Tensor([])

	tqdm_test_loader = tqdm(test_dataloader, desc=f"Evaluation: ", leave=False)
	
	avg_loss = 0
	with torch.no_grad():
		for step_num, batch_data in enumerate(tqdm_test_loader):

			tokens = batch_data["tokens"].to(device) 
			masks  = batch_data["attention_mask"].to(device)
			labels = batch_data["label"].to(device)

			logits = model(tokens, masks)
			loss = criterion(logits, labels)
			avg_loss += loss.item()
			numpy_logits = logits.cpu().detach()
			predicted = torch.cat((predicted, torch.argmax(numpy_logits, dim = 1)))
			all_logits  = torch.cat((all_logits, numpy_logits))
	
	average_loss = avg_loss / len(test_dataloader)
	print("average loss: ", average_loss)
	print()
	
	return predicted.tolist(), average_loss;

def calculate_metrics(test_dataloader, predicted):
	accuracy = accuracy_score(test_dataloader.dataset["label"], predicted)
	f1 = f1_score(test_dataloader.dataset["label"], predicted, average='weighted')  
	precision = precision_score(test_dataloader.dataset["label"], predicted, average='weighted')  
	recall = recall_score(test_dataloader.dataset["label"], predicted, average='weighted')
	mcc = matthews_corrcoef(test_dataloader.dataset["label"], predicted)
	conf_mat = confusion_matrix(test_dataloader.dataset["label"], predicted)

	metrics_dict = {
		"accuracy": accuracy,
		"f1": f1,
		"precision": precision,
		"recall": recall,
		"mcc": mcc,
		"conf_mat": conf_mat
	
	}
	return metrics_dict

def metrics_to_str(metrics):
	acc_str = f"Accuracy: {metrics['accuracy']:.3f}"
	f1_str = f"F1: {metrics['f1']:.3f}"
	prec_str = f"Precision: {metrics['precision']:.3f}"
	rec_str = f"Recall: {metrics['recall']:.3f}"
	mcc_str = f"MCC: {metrics['mcc']:.3f}"
	conf_mat_str = f"Confusion matrix:\n {metrics['conf_mat']}"

	return f"{acc_str}\n{f1_str}\n{prec_str}\n{rec_str}\n{mcc_str}\n{conf_mat_str}"