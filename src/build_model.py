from spiking_eegnet import SpikingEEGNet

def build_model(meta, device, **model_kwargs):
    model = SpikingEEGNet(
        num_classes=meta["n_classes"],
        num_channels=meta["n_channels"],
        num_samples=meta["n_samples"],
        **model_kwargs,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")
    return model