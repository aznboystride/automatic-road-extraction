def nonlinear(inplace=True):
    return nn.ReLU(inplace=inplace)

def norm(features):
    return nn.BatchNorm2d(features)

class DecoderBlock(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.in_ = in_
        self.block = nn.Sequential(
            nn.Conv2d(in_, in_//4, kernel_size=1),
            nn.BatchNorm2d(in_//4),
            nonlinear(),
            
            nn.ConvTranspose2d(in_//4, out, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out),
            nonlinear(),
        )

    def forward(self, t):
        return self.block(t)
