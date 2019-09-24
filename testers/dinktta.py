import cv2, numpy as np
from torch.autograd import Variable as V


class dinktta:
    def __init__(self, net, batchsize=1):
        self.net = net
        self.batch = batchsize

    def __call__(self, inputs):
        mask = self.test_one_img_from_path_2(inputs)
        mask[mask>4.0] = 255
        mask[mask<=4.0] = 0
        mask = np.concatenate([mask[:,:,None],mask[:,:,None],mask[:,:,None]],axis=2)
        return mask.astype(np.uint8) 

    def test_one_img_from_path_1(self, path):
        img = cv2.imread(path)#.transpose(2,0,1)[None]
                    
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.concatenate([img1,img2])
        img4 = np.array(img3)[:,:,::-1]
        img5 = np.concatenate([img3,img4]).transpose(0,3,1,2)
        img5 = np.array(img5, np.float32)/255.0 * 3.2 -1.6
        img5 = V(torch.Tensor(img5).cuda())

        mask = self.net.forward(img5).squeeze().cpu().data.numpy()#.squeeze(1)
        mask1 = mask[:4] + mask[4:,:,::-1]
        mask2 = mask1[:2] + mask1[2:,::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1,::-1]

        return mask3

    def test_one_img_from_path_2(self, inputs):
        img = inputs.squeeze(0).cpu().data.numpy()
        img = np.transpose(img, (1,2,0))
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.concatenate([img1,img2])
        img4 = np.array(img3)[:,:,::-1]
        img5 = img3.transpose(0,3,1,2)
        img5 = np.array(img5, np.float32)/255.0 * 3.2 -1.6
        img5 = V(torch.Tensor(img5).cuda())
        img6 = img4.transpose(0,3,1,2)
        img6 = np.array(img6, np.float32)/255.0 * 3.2 -1.6
        img6 = V(torch.Tensor(img6).cuda())

        maska = self.net.forward(img5).squeeze().cpu().data.numpy()#.squeeze(1)
        maskb = self.net.forward(img6).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:,:,::-1]
        mask2 = mask1[:2] + mask1[2:,::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1,::-1]

        return mask3
