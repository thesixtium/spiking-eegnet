def train_one_epoch(model, loader, optimizer, criterion, device, n_steps):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        spk = model(xb, num_steps=n_steps)
        # Average logits over timesteps before loss
        logits = spk.mean(0)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)