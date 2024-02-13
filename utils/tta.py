from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm


class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        return torch.mean(-(x.softmax(1) * x.log_softmax(1)).sum(1), 0)


def train_on_source_dataset(model, dataset, args):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.n_epochs)
    criterion = dataset.get_source_loss()
    model.train()
    train_set, test_set = dataset.get_source_dataset()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    for e in range(model.args.n_epochs):
        with tqdm(total=len(train_loader), desc=f"[EP{e}] Training on source dataset") as pbar:
            for i, data in enumerate(train_loader):
                if args.debug_mode and i > model.get_debug_iters():
                    break
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device, dtype=torch.long)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
                pbar.update()
                pbar.set_postfix({'loss': loss.item(), 'lr': opt.param_groups[0]['lr']})
        sched.step()
        total = 0
        correct = 0
        model.eval()
        with torch.no_grad():
            for i, data in tqdm(enumerate(test_loader), desc=f"[EP{e}] Testing on source dataset", total=len(test_loader)):
                if args.debug_mode and i > model.get_debug_iters():
                    break
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device, dtype=torch.long)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"EP{e} acc = {correct/total:.2%}")
    Path(f'./data/checkpoints').mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f'./data/checkpoints/{args.dataset}_{e}_source.pt')
    print(f"Source task accuracy: {correct/total:.2%}, saved to ./data/checkpoints/{args.dataset}_{e}_source.pt")
