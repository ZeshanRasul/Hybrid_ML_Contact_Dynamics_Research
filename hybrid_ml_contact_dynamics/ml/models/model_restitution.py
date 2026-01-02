import torch

class RestitutionPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(13434, 1)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(1, 1000)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.tensor):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x
    

def main():
    pass
