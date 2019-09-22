import os, cv2, numpy as np

class Loader:

    def __init__(self, root, test=False, augmentation=None):
        self.test = test
        self.root = root
        self.imagelist = list(filter(lambda x: x.find('sat')!=-1, os.listdir(root)))
        self.trainlist = list(map(lambda x: x[:-8], self.imagelist))
        self.augmentation = augmentation

    def load(self, index):
        index = self.trainlist[index]
        img = cv2.imread(os.path.join(self.root, '{}_sat.jpg'.format(index)))
        mask = cv2.imread(os.path.join(self.root, '{}_mask.png'.format(index)), cv2.IMREAD_GRAYSCALE)
        if self.augmentation:
            img, mask = self.augmentation(img, mask)

        mask = np.expand_dims(mask, axis=2)
        img = np.array(img, np.float32).transpose(2,0,1)/255.0
        mask = np.array(mask, np.float32).transpose(2,0,1)/255.0
        mask[mask>=0.5] = 1
        mask[mask<=0.5] = 0
        return img, mask

    def tload(self, index):
        index = self.trainlist[index]
        path = os.path.join(self.root, '{}_sat.jpg'.format(index))
        img = cv2.imread(path)

        img = np.array(img, np.float32).transpose(2,0,1)/255.0
        return os.path.basename(path.replace('sat.jpg', 'mask.png')), img

    def __len__(self):
        return len(self.trainlist)

    def __call__(self, index):
        if self.test:
            return self.tload(index)
        else:
            return self.load(index)
