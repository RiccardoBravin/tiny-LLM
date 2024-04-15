import torch
from tqdm import tqdm


# Print the model size
def print_model_size(model):
  param_size = 0
  param_count = 0
  for param in model.parameters():
    param_size += param.nelement() * param.element_size()
    param_count += param.nelement()
  buffer_size = 0
  for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

  size_all_mb = (param_size + buffer_size) / 1024**2
  print('Model params: {:.3f}M'.format(param_count/1e6))
  print('Model size: {:.3f}MB'.format(size_all_mb))



def pad_sequences(ls, maxlen=512, truncating="post", padding="post", dtype="int"):
  res = []
  for innermost in ls[:]:
    pad_len = max(0, maxlen - len(innermost))
    res.append(innermost + [0] * pad_len)

  return res
  

  

def trainer(model, train_dataloader, lr, epochs):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  criterion = torch.nn.CrossEntropyLoss()
  criterion.to(device);

  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  for epoch in range(epochs):
    model.train()
    train_loss = 0
    tqdm_train_loader = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", leave=False)

    for step_num, batch_data in enumerate(tqdm_train_loader):

        token_ids, masks, labels = tuple(t.to(device) for t in batch_data)

        logits = model(token_ids)

        batch_loss = criterion(logits, labels)
        train_loss += batch_loss.item()

        model.zero_grad()
        batch_loss.backward()


        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        log_step = 50
        if step_num % log_step == (log_step - 1):
          tqdm_train_loader.set_postfix(loss = train_loss / log_step)
          train_loss = 0


def evaluator(model, test_dataloader):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  criterion = torch.nn.CrossEntropyLoss()
  criterion.to(device);

  model.eval()
  predicted = torch.Tensor([])
  all_logits = torch.Tensor([])

  tqdm_test_loader = tqdm(test_dataloader, desc=f"Evaluation: ", leave=False)

  with torch.no_grad():
      for step_num, batch_data in enumerate(tqdm_test_loader):

          token_ids, masks, labels = tuple(t.to(device) for t in batch_data)

          logits = model(token_ids)
          loss = criterion(logits, labels)
          numpy_logits = logits.cpu().detach()
          predicted = torch.cat((predicted, torch.argmax(numpy_logits, dim = 1)))
          all_logits  = torch.cat((all_logits, numpy_logits))
  print()

  return predicted.tolist();