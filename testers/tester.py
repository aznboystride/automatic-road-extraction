import cv2

class tester:
    def __init__(self, net, batchsize=1):
        self.net = net
        self.batch = batchsize

    def __call__(self, inputs):
        image = self.net(inputs.cuda()).squeeze().squeeze().cpu().data.numpy()
        image[image>0.5] = 255
        image[image<=0.5] = 0
        return cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
