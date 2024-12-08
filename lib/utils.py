
from lib.colors import FOREGROUND_COLORS, RESET
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
from scipy.stats import spearmanr
from transformers import EvalPrediction, TrainerCallback
from tqdm import tqdm
import torch
import os

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



def compute_metrics(p: EvalPrediction):
    try:

        preds = p.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='weighted', zero_division=0)
        mcc = matthews_corrcoef(p.label_ids, preds)
        acc = accuracy_score(p.label_ids, preds)
        
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'mcc': mcc,
        }
    except Exception as e:

        preds = p.predictions
        scc = spearmanr(p.label_ids, preds)[0]
        return {
            'scc': scc
        }



class CustomPrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs, **kwargs):
        control.should_log = False
        if state.is_world_process_zero:
            out_logs = {}
            epoch = int(logs['epoch'])
            is_eval = False

            if 'loss' in logs:
                out_logs['loss'] = f"{logs['loss']:.6f}"

            if 'learning_rate' in logs:
                out_logs['learning_rate'] = f"{logs['learning_rate']}"


            if 'eval_loss' in logs:
                out_logs['eval_loss'] = f"{logs['eval_loss']:.6f}"
                is_eval = True

            if 'eval_mlm_loss' in logs:
                out_logs['eval_mlm_loss'] = f"{logs['eval_mlm_loss']:.6f}"
            
            if 'eval_nsp_loss' in logs:
                out_logs['eval_nsp_loss'] = f"{logs['eval_nsp_loss']:.6f}"

            if 'eval_accuracy' in logs:
                out_logs['eval_accuracy'] = f"{logs['eval_accuracy']*100:.2f}"

            if 'eval_f1' in logs:
                out_logs['eval_f1'] = f"{logs['eval_f1']*100:.2f}"

            if 'eval_mcc' in logs:
                out_logs['eval_mcc'] = f"{logs['eval_mcc']*100:.2f}"

            if 'eval_scc' in logs:
                out_logs['eval_scc'] = f"{logs['eval_scc']*100:.2f}"

            
            if out_logs:
                if is_eval:
                    output = f"{FOREGROUND_COLORS['BrightGreen']}[Eval {epoch}]\t"
                else:
                    output = f"{FOREGROUND_COLORS['Green']}[Train {epoch}]\t"

                output += ' '.join([f'{key}: {str(value).ljust(10)}' for key, value in out_logs.items()])
                output += RESET

                tqdm.write(output)

    def on_epoch_begin(self, args, state, control, **kwargs):
        if "finetuning" in args.run_name:
            tqdm.write(f"{FOREGROUND_COLORS["BrightMagenta"]}")
        elif "quantized" in args.run_name:
            tqdm.write(f"{FOREGROUND_COLORS["BrightCyan"]}")
        else:
            tqdm.write(f"{FOREGROUND_COLORS["BrightYellow"]}")

        return super().on_epoch_begin(args, state, control, **kwargs)

class CustomLoggerCallback(TrainerCallback):
     def on_log(self, args, state, control, logs, **kwargs):
        
        if state.is_world_process_zero:

            step = int(state.global_step)

            train_loss = None
            lr = None
            eval_loss = None

            
            if 'loss' in logs:
                train_loss = logs['loss']

            if 'eval_loss' in logs:
                eval_loss = logs['eval_loss']

    
            if eval_loss:
                output = f"{step}: {eval_loss}\n"
                #write to file ./results/{args.output_dir}/eval_loss.txt
                with open(f"{args.output_dir}/eval_loss.txt", "a") as f:
                    f.write(output)
            else:
                output = f"{step}: {train_loss}\n"
                #write to file ./results/{args.output_dir}/train_loss.txt
                with open(f"{args.output_dir}/train_loss.txt", "a") as f:
                    f.write(output)

            

def save_model_score(metrics:dict, output_dir:str, filename:str):


    #if directory does not exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_dir+filename, "a") as f:
        f.write("-------------------------------\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
        f.write("\n\n")


def make_score_mask(input_tensor):
    """
    Args:
        input_tensor: the input tensor of shape (batch_size, max_len)
    Returns:
        Tensor of shape (batch_size, max_len, max_len)
    """
    # Get batch size and sentence length
    batch_size, max_length = input_tensor.shape

    # Initialize the output tensor with zeros
    output_tensor = torch.zeros((batch_size, max_length, max_length), dtype=input_tensor.dtype, device=input_tensor.device)

    # Fill the output tensor according to the input tensor values
    for i in range(batch_size):
        sentence_len = (input_tensor[i] != 0).sum()  # Get the non-zero length (count of 1s)
        output_tensor[i, :sentence_len, :] = input_tensor[i]

    return output_tensor