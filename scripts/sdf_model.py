import torch
import torch.nn as nn


class SDFModel(nn.Module):

    def __init__(self, num_layers, skip_connections, latent_size, inner_dim=512, output_dim=1):
        super().__init__()
        self.num_layers = num_layers
        self.skip_connections = skip_connections
        self.latent_size = latent_size
        input_dim = latent_size + 3
        self.skip_tensor_dim = input_dim
        num_extra = 2 if skip_connections and num_layers >= 8 else 1
        layers = []
        for _ in range(num_layers - num_extra):
            layers.append(nn.Sequential(nn.utils.weight_norm(nn.Linear(input_dim, inner_dim)), nn.ReLU()))
            input_dim = inner_dim
        self.net = nn.Sequential(*layers)
        self.final_layer = nn.Sequential(nn.Linear(inner_dim, output_dim), nn.Tanh())
        self.skip_layer = nn.Sequential(nn.Linear(inner_dim, inner_dim - self.skip_tensor_dim), nn.ReLU())

    def forward(self, x):
        x_in = x.detach()
        if self.skip_connections and self.num_layers >= 5:
            for i in range(3):
                x = self.net[i](x)
            x = self.skip_layer(x)
            x = torch.hstack((x, x_in))
            for i in range(self.num_layers - 5):
                x = self.net[3 + i](x)
        else:
            x = self.net(x)
        return self.final_layer(x)


def load_decoder(run_dir, device):
    """Load a decoder + its settings dict from a runs_sdf/<timestamp> folder."""
    import yaml
    from pathlib import Path
    run_dir = Path(run_dir)
    with open(run_dir / 'settings.yaml') as f:
        cfg = yaml.safe_load(f)
    model = SDFModel(cfg['num_layers'], cfg['skip_connections'],
                     cfg['latent_size'], cfg['inner_dim']).to(device)
    model.load_state_dict(torch.load(run_dir / 'weights.pt',
                                     map_location=device, weights_only=False))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg
