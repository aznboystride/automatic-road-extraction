import os, cv2, numpy as np
import glob

class Loader:

    def __init__(self, root, test=False, augmentation=None):
        self.test = test
        self.root = root
        # self.imagelist = list(filter(lambda x: x.find('sat')!=-1, os.listdir(root)))
        dataset_sat_glob = glob.glob(f'{root}/{"test" if test else "train"}/sat/*sat.jpg', recursive=True)
        dataset_map_glob = glob.glob(f'{root}/{"test" if test else "train"}/sat/*15.png', recursive=True)

        self.imageSatPaths = dataset_sat_glob + dataset_map_glob
        self.imageSatNumbers = list(map(lambda x: os.path.basename(x).split('_')[0], self.imageSatPaths))
        print(f"[Loader] root path:               {root}")
        print(f"[Loader] length of imageSatPaths:       {len(self.imageSatPaths)}")
        print(f"[Loader] Example of imageSatPaths[0]:   {self.imageSatPaths[0]}")
        print(f"[Loader] length of imageSatNumbers:     {len(self.imageSatNumbers)}")
        print(f"[Loader] Example of imageSatNumbers[0]: {self.imageSatNumbers[0]}")
        self.augmentation = augmentation

    def load(self, index):
        index = self.imageSatNumbers[index]
        # img = cv2.imread(os.path.join(self.root, '{}_sat.jpg'.format(index)))
        # mask = cv2.imread(os.path.join(self.root, '{}_mask.png'.format(index)), cv2.IMREAD_GRAYSCALE)
        sat_dir = os.path.dirname(self.imageSatPaths[0])
        sat_path = os.path.join(sat_dir, f'{index}_sat.jpg')
        if not os.path.exists(sat_path):
            sat_path = sat_path.replace('sat.jpg', '15.png')
        if not os.path.exists(sat_path):
            print(f"[Loader ERROR]: sat path does not exists: {sat_path}")
            print(f"Exiting...")
            exit(1)

        mask_dir = os.path.dirname(self.imageSatPaths[0]).replace('sat', 'map')
        mask_path = os.path.join(mask_dir, f'{index}_mask.png')
        if not os.path.exists(mask_path):
            mask_path = mask_path.replace('mask', '15')
        if not os.path.exists(mask_path):
            print(f"[Loader ERROR]: mask path does not exists: {mask_path}")
            print(f"Exiting...")
            exit(1)

        img = cv2.imread(sat_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if self.augmentation:
            img, mask = self.augmentation(img, mask)

        mask = np.expand_dims(mask, axis=2)
        img = np.array(img, np.float32).transpose(2,0,1)/255.0
        mask = np.array(mask, np.float32).transpose(2,0,1)/255.0
        mask[mask>=0.5] = 1
        mask[mask<=0.5] = 0
        return img, mask

    def tload(self, index):
        index = self.imageSatNumbers[index]
        path = os.path.join(self.root, '{}_sat.jpg'.format(index))
        img = cv2.imread(path)

        img = np.array(img, np.float32).transpose(2,0,1)/255.0
        return os.path.basename(path.replace('sat.jpg', 'mask.png')), img

    def __len__(self):
        return len(self.imageSatNumbers)

    def __call__(self, index):
        if self.test:
            return self.tload(index)
        else:
            return self.load(index)
