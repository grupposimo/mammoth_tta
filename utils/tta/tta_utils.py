from pathlib import Path

import torch
from tqdm import tqdm


def train_on_source_dataset(model, dataset, args):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.n_epochs)
    criterion = dataset.get_source_loss()
    train_set, test_set = dataset.get_source_dataset()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    for e in range(model.args.n_epochs):
        model.train()
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
        # sched.step()
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

        p = Path(f'./data/checkpoints/no_sched_{args.dataset}_{e}_source.pt').resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), p)
        print(f"Source task accuracy: {correct/total:.2%}, saved to {p}")


def sanity_check(model, dataset):
    model_status = model.training
    total = 0
    correct = 0
    if hasattr(dataset, 'get_source_dataset'):
        model.eval()
        _, test_set = dataset.get_source_dataset()
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)
        with torch.no_grad():
            for data in tqdm(test_loader, desc=f"Sanity checking", total=len(test_loader)):
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device, dtype=torch.long)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        model.train(model_status)
        print(f"Sanity check accuracy:  {correct / total:.2%}")
    else:
        print("No source dataset, skipping sanity check")
