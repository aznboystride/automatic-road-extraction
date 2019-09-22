

class tester:
    def __init__(self, net, batchsize=1):
        self.net = net
        self.batch = batchsize

    def __call__(self, inputs):
        image = self.net(inputs.cuda()).squeeze().squeeze().cpu().data.numpy()
        return image
