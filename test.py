import torch

from pytorch_concurrent_dataloader import DataLoader
from torch.utils.data import TensorDataset

def main():
    dataset = TensorDataset(torch.randn(512))
    dataloader = DataLoader(dataset, batch_size=32, num_workers=1)
    for i, _ in enumerate(dataloader):
        print(f"loaded batch {i}")


if __name__ == "__main__":
    main()