import torch
from torch import nn


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        output_size: int,
        n_blocks: int = 2,
        base_features: int = 256,
    ) -> None:
        super().__init__()

        self.input = self._block(latent_dim, base_features)
        self.feature_extractor = self._make_block_chain(n_blocks, base_features)
        self.output = nn.Linear(base_features * 2**n_blocks, output_size)

    def _block(self, in_features: int, out_features: int) -> nn.Sequential:
        layers = []

        layers.append(nn.Linear(in_features, out_features, bias=False))
        layers.append(nn.BatchNorm1d(out_features))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _make_block_chain(self, n_blocks: int, in_features: int) -> nn.Sequential:
        blocks = []

        for _ in range(n_blocks):
            blocks.append(self._block(in_features, in_features * 2))
            in_features = in_features * 2

        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = self.feature_extractor(x)
        x = self.output(x)

        return x


class Critic(nn.Module):
    def __init__(
        self, input_size: int, n_blocks: int = 2, base_features: int = 256
    ) -> None:
        super().__init__()

        self.input = self._block(input_size, base_features * 2**n_blocks)
        self.feature_extractor = self._make_block_chain(n_blocks, base_features)
        self.output = nn.Linear(base_features, 1)

    def _block(
        self, in_features: int, out_features: int, use_norm: bool = True
    ) -> nn.Sequential:
        layers = []

        layers.append(nn.Linear(in_features, out_features, bias=not use_norm))
        if use_norm:
            layers.append(nn.LayerNorm(out_features))
        layers.append(nn.LeakyReLU(0.2))

        return nn.Sequential(*layers)

    def _make_block_chain(self, n_blocks: int, in_features: int) -> nn.Sequential:
        blocks = []

        in_channels = in_features * 2**n_blocks

        for _ in range(n_blocks):
            blocks.append(self._block(in_channels, in_channels // 2))
            in_channels = in_channels // 2

        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = self.feature_extractor(x)
        x = self.output(x)

        return x


def test():
    batch_size = 32
    latent_dim = 128
    gene_expr_size = 2000

    gen = Generator(latent_dim, gene_expr_size)
    critic = Critic(gene_expr_size)

    noise = torch.randn(batch_size, latent_dim)
    real = torch.randn(batch_size, gene_expr_size)

    fake = gen(noise)
    print(fake.shape)

    pred = critic(real)
    print(pred.shape)


if __name__ == "__main__":
    test()
