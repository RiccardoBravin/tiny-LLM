
import sentencepiece as spm
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn 

from torch.utils.data import Dataset

from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

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


def train_sp(text_train, vocab_size, dataset_name):

	str_text_files = '**/' + dataset_name + '_text_*.txt'

	#save the dataset to a file for sentecepiece training
	paths = [str(x) for x in Path('./stpiece').glob(str_text_files)]
	if paths == []:
		text_data = []
		file_count = 0
		for sample in tqdm(text_train):
			text_data.append(sample)
			# once we hit the 10K mark, save to file
			if len(text_data) == 10000:
				with open(f'./stpiece/{dataset_name}_text_{file_count}.txt', 'w', encoding='utf-8') as fp:
					fp.write('\n'.join(text_data))
				text_data = []
				file_count += 1
		# save the remaining data
		with open(f'./stpiece/{dataset_name}_text_{file_count}.txt', 'w', encoding='utf-8') as fp:
			fp.write('\n'.join(text_data))
		

	paths = [str(x) for x in Path('./stpiece').glob(str_text_files)]
	#check if sentencepiece model already exists and if not train it
	if not len([x for x in Path('./stpiece').glob('**/' + dataset_name + '_'+ str(vocab_size) + "*")]) == 2:
		spm.SentencePieceTrainer.train(input=paths, model_prefix="./stpiece/" + dataset_name + '_' + str(vocab_size), vocab_size=vocab_size)

	#load sentencepiece model
	sp = spm.SentencePieceProcessor()
	sp.load("./stpiece/" + dataset_name + '_' + str(vocab_size) + ".model")

	return sp




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


class ds(Dataset):
	def __init__(self, tokens, labels):
		self.tokens = tokens
		self.labels = labels

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		return self.tokens[idx], self.labels[idx] 

def pad_sequences(ls, maxlen=512, truncating="post", padding="post", dtype="int"):
	res = []
	for innermost in ls[:]:
		pad_len = max(0, maxlen - len(innermost))
		res.append(innermost + [0] * pad_len)

	return res
	


def trainer(model, train_dataloader, val_dataloader, lr, epochs):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	criterion = torch.nn.CrossEntropyLoss()
	criterion.to(device);

	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	scheduler = ReduceLROnPlateau(optimizer, 'min')
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs*10)//2, eta_min=1e-6)

	log_step = len(train_dataloader) // 10

	for epoch in range(epochs):
		tqdm.write(f"Epoch {epoch+1}/{epochs}")
		
		model.train()
		train_loss = 0
		tqdm_train_loader = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

		for step_num, batch_data in enumerate(tqdm_train_loader):

			token_ids, labels = tuple(t.to(device) for t in batch_data)

			logits = model(token_ids)

			batch_loss = criterion(logits, labels)
			if step_num == 0:
				train_loss = batch_loss.item()
			else:
				train_loss = 0.9 * train_loss + 0.1 * batch_loss.item()

			model.zero_grad()
			batch_loss.backward()


			#torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optimizer.step()
			

			loss_str = "{:.4f}".format(train_loss)
			tqdm_train_loader.set_postfix(loss = loss_str)			
			if step_num % log_step == (log_step - 1):
				model.eval()
				val_accuracy = []
				val_loss = 0
				for x, y in val_dataloader:
					x, y = x.to(device), y.to(device)

					guess = model(x)
					val_loss += criterion(guess, y).item()
					val_accuracy.append((torch.argmax(guess, dim=1) == y).item())	
				
				val_accuracy = sum(val_accuracy) / len(val_accuracy)
				val_loss = val_loss / len(val_dataloader)
				scheduler.step()
				tqdm.write(f"Validation loss: {val_loss:.3f}, Validation accuracy: {val_accuracy:.3f}, Lr: {scheduler.get_last_lr()[0]:.6f}")
				model.train()




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

			token_ids, labels = tuple(t.to(device) for t in batch_data)

			logits = model(token_ids)
			loss = criterion(logits, labels)
			avg_loss += loss.item()
			numpy_logits = logits.cpu().detach()
			predicted = torch.cat((predicted, torch.argmax(numpy_logits, dim = 1)))
			all_logits  = torch.cat((all_logits, numpy_logits))
	
	average_loss = avg_loss / len(test_dataloader)
	print("average loss: ", average_loss)
	print()
	
	return predicted.tolist(), average_loss;