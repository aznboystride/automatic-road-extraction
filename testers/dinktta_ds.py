import cv2, numpy as np
from torch.autograd import Variable as V
import torch
import os


class dinktta_ds:
    def __init__(self, net, batchsize=1):
        self.net = net
        self.batch = batchsize

    def __call__(self, inputs):
        mask = self.batch1(inputs) if self.batch == 1 else\
                self.batch2(inputs) if self.batch == 2 else\
                self.batch4(inputs) if self.batch == 4 else\
                self.batch8(inputs)
        mask[mask>=4.0] = 255
        mask[mask<4.0] = 0
        mask = np.concatenate([mask[:,:,None],mask[:,:,None],mask[:,:,None]],axis=2)
        return mask.astype(np.uint8)

    def batch1(self, path):
        #print(1)
        img = cv2.imread(path)
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.concatenate([img1,img2])
        img4 = np.array(img3)[:,:,::-1]
        img5 = np.concatenate([img3,img4]).transpose(0,3,1,2)
        img5 = np.array(img5, np.float32)/255.0
        img5 = V(torch.Tensor(img5).cuda())

        mask = self.net.forward(img5).squeeze().cpu().data.numpy()#.squeeze(1)
        mask1 = mask[:4] + mask[4:,:,::-1]
        mask2 = mask1[:2] + mask1[2:,::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1,::-1]

        return mask3

    def batch2(self, path):
        #print(2)
        img = cv2.imread(path)
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.concatenate([img1,img2])
        img4 = np.array(img3)[:,:,::-1]
        img5 = img3.transpose(0,3,1,2)
        img5 = np.array(img5, np.float32)/255.0
        img5 = V(torch.Tensor(img5).cuda())
        img6 = img4.transpose(0,3,1,2)
        img6 = np.array(img6, np.float32)/255.0
        img6 = V(torch.Tensor(img6).cuda())

        maska = self.net.forward(img5).squeeze().cpu().data.numpy()#.squeeze(1)
        maskb = self.net.forward(img6).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:,:,::-1]
        mask2 = mask1[:2] + mask1[2:,::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1,::-1]

        return mask3

    def batch4(self, path):
        img = cv2.imread(path)
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.array(img1)[:,:,::-1]
        img4 = np.array(img2)[:,:,::-1]

        img1 = img1.transpose(0,3,1,2)
        img2 = img2.transpose(0,3,1,2)
        img3 = img3.transpose(0,3,1,2)
        img4 = img4.transpose(0,3,1,2)

        img1 = V(torch.Tensor(np.array(img1, np.float32)/255.0).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32)/255.0).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32)/255.0).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32)/255.0).cuda())

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:,::-1] + maskc[:,:,::-1] + maskd[:,::-1,::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1,::-1]

        return mask2

    def batch8(self, path):
        img = cv2.imread(path)
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.array(img1)[:,:,::-1]
        img4 = np.array(img2)[:,:,::-1]

        img1 = img1.transpose(0,3,1,2)
        img2 = img2.transpose(0,3,1,2)
        img3 = img3.transpose(0,3,1,2)
        img4 = img4.transpose(0,3,1,2)

        img1 = V(torch.Tensor(np.array(img1, np.float32)/255.0).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32)/255.0).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32)/255.0).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32)/255.0).cuda())

        o1 = self.net.forward(img1) 
        o2 = self.net.forward(img2)
        o3 = self.net.forward(img3)
        o4 = self.net.forward(img4)

        maska = (o1[0]).squeeze().cpu().data.numpy()
        maskb = (o2[0]).squeeze().cpu().data.numpy()
        maskc = (o3[0]).squeeze().cpu().data.numpy()
        maskd = (o4[0]).squeeze().cpu().data.numpy()

        '''maska = ((o1[0]+o1[1]+o1[2]+o1[3]+o1[4])/5).squeeze().cpu().data.numpy()
        maskb = ((o2[0]+o2[1]+o2[2]+o2[3]+o2[4])/5).squeeze().cpu().data.numpy()
        maskc = ((o3[0]+o3[1]+o3[2]+o3[3]+o3[4])/5).squeeze().cpu().data.numpy()
        maskd = ((o4[0]+o4[1]+o4[2]+o4[3]+o4[4])/5).squeeze().cpu().data.numpy()'''

        '''maska = ((o1[0]+o1[1]+o1[2]+o1[3])/4).squeeze().cpu().data.numpy()
        maskb = ((o2[0]+o2[1]+o2[2]+o2[3])/4).squeeze().cpu().data.numpy()
        maskc = ((o3[0]+o3[1]+o3[2]+o3[3])/4).squeeze().cpu().data.numpy()
        maskd = ((o4[0]+o4[1]+o4[2]+o4[3])/4).squeeze().cpu().data.numpy()'''

        mask1 = maska + maskb[:,::-1] + maskc[:,:,::-1] + maskd[:,::-1,::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1,::-1]

        return mask2
