
import sentencepiece as spm
from pathlib import Path
import tqdm

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


def train_sp(train_data, vocab_size, tokens):

    #save the dataset to a file for sentecepiece training
    paths = [str(x) for x in Path('./stpiece').glob('**/text_*.txt')]
    if paths == []:
        text_data = []
        file_count = 0
        for sample in tqdm.tqdm(train_data["question"]):
            text_data.append(sample)
            # once we hit the 10K mark, save to file
            if len(text_data) == 100000:
                with open(f'./stpiece/text_{file_count}.txt', 'w', encoding='utf-8') as fp:
                    fp.write('\n'.join(text_data))
                text_data = []
                file_count += 1
        
        for sample in tqdm.tqdm(train_data["response"]):
            text_data.append(sample)
            # once we hit the 10K mark, save to file
            if len(text_data) == 100000:
                with open(f'./stpiece/text_{file_count}.txt', 'w', encoding='utf-8') as fp:
                    fp.write('\n'.join(text_data))
                text_data = []
                file_count += 1


    paths = [str(x) for x in Path('./stpiece').glob('**/text_*.txt')]
    #check if sentencepiece model already exists and if not train it
    if not len([x for x in Path('./stpiece').glob('**/m_*'+ str(vocab_size) + "*")]) == 2:
        spm.SentencePieceTrainer.train(input=paths, model_prefix="./stpiece/m_orca_dictsz_" + str(vocab_size), vocab_size=vocab_size, bos_id=tokens['bos'], eos_id=tokens['eos'], pad_id=tokens['pad'], unk_id=tokens['unk'], user_defined_symbols=["<mask>"])

    #load sentencepiece model
    sp = spm.SentencePieceProcessor()
    sp.load("./stpiece/m_orca_dictsz_" + str(vocab_size) + ".model")

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

