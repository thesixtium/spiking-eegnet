def aggregate_logits(spk, mem, readout_mode):
    """
    Aggregate spiking network outputs into a single logit tensor.

    Parameters
    ----------
    spk : Tensor  (num_steps, batch, num_classes)   — classifier outputs from spike path
    mem : Tensor  (num_steps, batch, filters, 1, T) — LIF membrane potentials before pool2
    readout_mode : str
        "spk_mean"  — average logits over all timesteps  (original behaviour)
        "spk_last"  — use only the last timestep's logits
        "spk_sum"   — sum logits over all timesteps
        "mem_last"  — pass last-step membrane potential through the classifier
                      NOTE: caller must pass the model so we can call model.classifier;
                      this variant receives pre-flattened mem logits already computed
                      inside forward(), so we just take the last step.
    """
    if readout_mode == "spk_mean":
        return spk.mean(0)
    elif readout_mode == "spk_last":
        return spk[-1]
    elif readout_mode == "spk_sum":
        return spk.sum(0)
    elif readout_mode == "mem_last":
        # mem[-1]: (batch, filters, 1, T) → flatten → (batch, flat)
        # The model's classifier maps flat → num_classes, same as the spike path.
        last_mem = mem[-1]                          # (batch, C, 1, T)
        return last_mem.flatten(1)                  # (batch, flat) — caller feeds to classifier
    else:
        raise ValueError(f"Unknown readout_mode: {readout_mode!r}. "
                         "Choose from: 'spk_mean', 'spk_last', 'spk_sum', 'mem_last'.")


def train_one_epoch(model, loader, optimizer, criterion, device, n_steps,
                    readout_mode="spk_mean"):
    """
    Train for one epoch.

    Parameters
    ----------
    readout_mode : str
        How to collapse the (num_steps, batch, num_classes) spike outputs into
        per-sample logits. See aggregate_logits() for options.
        'mem_last' routes the final LIF membrane state through model.classifier
        instead of using the spike-path logits.
    """
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()

        spk, mem = model(xb, num_steps=n_steps)

        if readout_mode == "mem_last":
            # mem[-1]: (batch, filters, 1, T) — flatten and pass through classifier
            logits = model.classifier(mem[-1].flatten(1))
        else:
            logits = aggregate_logits(spk, mem, readout_mode)

        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)