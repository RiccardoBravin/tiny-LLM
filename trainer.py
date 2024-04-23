import numpy as np
import torch
import tqdm

from sklearn.metrics import matthews_corrcoef


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr



class BERTTrainer:
    def __init__(self, model, train_dataloader, test_dataloader=None, lr= 1e-4, weight_decay=0.01, betas=(0.9, 0.999), warmup_steps=10000, log_freq=100, device='cpu'):

        self.device = device
        self.model = model
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(
            self.optim, self.model.bert.d_model, n_warmup_steps=warmup_steps
            )

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion_llm = torch.nn.NLLLoss(ignore_index=0)
        self.criterion_cls = torch.nn.NLLLoss()

        self.log_freq = log_freq
        print("Total Parameters of LLM:", sum([p.nelement() for p in self.model.parameters()]))
    
    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        
        avg_loss_lm = 0.0
        avg_loss_next = 0.0
        total_correct = 0
        total_element = 0
        
        mode = "train" if train else "test"

        # progress bar
        data_iter = tqdm.tqdm(
            enumerate(data_loader),
            desc="EP_%s:%d" % (mode, epoch),
            total=len(data_loader),
            
        )

        for i, data in data_iter:

            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the next_sentence_prediction and masked_lm model
            next_sent_output, mask_lm_output = self.model.forward(data["bert_input"], data["segment_label"])
            
            # 2-1. NLL(negative log likelihood) loss of is_next classification result     
            next_loss = self.criterion_cls(next_sent_output, data["is_next"])

        
            # 2-2. NLLLoss of predicting masked token word
            # transpose to (m, vocab_size, seq_len) vs (m, seq_len)
            # criterion(mask_lm_output.view(-1, mask_lm_output.size(-1)), data["bert_label"].view(-1))
            mask_loss = self.criterion_llm(mask_lm_output.transpose(1, 2), data["bert_label"])

            # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
            loss = next_loss + mask_loss

            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            # next sentence prediction accuracy

            
            correct_next = next_sent_output.argmax(dim=-1)
            avg_loss_next += next_loss.item()
            avg_loss_lm += mask_loss.item()
            total_correct += correct_next.eq(data["is_next"]).sum().item()
            total_element += len(data["is_next"])

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss_next": round(avg_loss_next / (i + 1), 3),
                "avg_loss_lm": round(avg_loss_lm / (i + 1), 3),
                "mcc_next": round(matthews_corrcoef(data["is_next"].cpu(), correct_next.cpu()), 3),
                "loss": round(loss.item(),3)
            }

            if train and i % self.log_freq == 0:
                data_iter.write(str(post_fix))
            elif not train and i % (self.log_freq//10) == 0:
                data_iter.write(str(post_fix))


        print(
            f"EP{epoch}, {mode}: \
            avg_loss_next={round(avg_loss_next / len(data_iter), 3)}, \
            avg_loss_lm={round(avg_loss_lm / len(data_iter), 3)}, \
            total_acc={round(total_correct * 100.0 / total_element, 3)}"
        ) 