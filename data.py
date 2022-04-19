from importmod import *
import pandas as pd
import random
from skimage.transform import rotate
from torch.utils.data import Dataset, DataLoader

class MyRotationTransform:
    def __init__(self,angles=[0,90,-90.180]):
        self.angles = angles
    def _r(self,x,angle):
        return rotate(x,angle,preserve_range=True)
    def __call__(self,x,y,z,g):
        angle = random.choice(self.angles)
        return self._r(x,angle),self._r(y,angle),self._r(z,angle),self._r(g,angle)

class r3dDataset(Dataset):
    def __init__(self,csv_file='r3d.csv',size=512,transform=None):
        self.df = pd.read_csv(csv_file)
        self.df2 = pd.read_csv('r3d2.csv')
        self.size = size
        self.transform = transform
        self.rotation = MyRotationTransform()
    def __len__(self):
        return self.df.shape[0]+self.df2.shape[0]
    def _getset(self,idx): 
        target = self.df if idx < self.df.shape[0] else self.df2
        idx = idx if idx < self.df.shape[0] else idx-self.df.shape[0]
        image = np.fromstring(target.loc[idx]['image'][1:-1],
                dtype=np.uint8,sep=', ').reshape(self.size,self.size,3)
        boundary = np.fromstring(target.loc[idx]['boundary'][1:-1],
                dtype=np.uint8,sep=', ').reshape(self.size,self.size)
        room = np.fromstring(target.loc[idx]['room'][1:-1],
                dtype=np.uint8,sep=', ').reshape(self.size,self.size)
        door = np.fromstring(target.loc[idx]['door'][1:-1],
                dtype=np.uint8,sep=', ').reshape(self.size,self.size)
        return image,boundary,room,door
    def __getitem__(self,idx):
        image,boundary,room,door = self._getset(idx)
        #image,boundary,room,door = self.rotation(image,boundary,room,door)
        if self.transform:
            image = self.transform(image.astype(np.float32)/255.0)
            boundary = self.transform(F.one_hot(
                torch.LongTensor(boundary),3).numpy())
            room = self.transform(F.one_hot(
                torch.LongTensor(room),9).numpy())
            door = self.transform(door)
        return image,boundary,room,door

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    DFPdataset = r3dDataset()
    image,boundary,room,door = DFPdataset[200]

    plt.subplot(2,2,1); plt.imshow(image)
    plt.subplot(2,2,2); plt.imshow(boundary)
    plt.subplot(2,2,3); plt.imshow(room)
    plt.subplot(2,2,4); plt.imshow(door)
    plt.show()

    breakpoint()
    
    gc.collect()

