import torch.nn as nn

class FakeModel(nn.Module) :
    def __init__(self, L, input_channels) :
        super().__init__()
        print('Be careful : this is a Fake model with no weights just to run classical EM')

    def forward(self, x) :
        return None
