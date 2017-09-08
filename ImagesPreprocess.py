
# coding: utf-8

# In[2]:

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import sqlite3
import traceback


# In[3]:

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#plt.ion()   # interactive mode


# In[4]:

path_to_db = '/Users/romina/Databases/aflw/data/aflw.sqlite'
path_to_imgs = '/Users/romina/Databases/aflw/data/flickr/'
path_to_positives = '/Users/romina/Documents/Masters/pos_training/'
path_to_tmp = '/Users/romina/Documents/Masters/tmp/'


# In[5]:

conn = sqlite3.connect(path_to_db)
c = conn.cursor()
c.execute("""select Faces.face_id, Faces.file_id,x,y,w,h
from FacePose 
join Faces on FacePose.face_id = Faces.face_id
join FaceRect on FacePose.face_id = FaceRect.face_id
where abs(FacePose.yaw * 180/3.14159) <= 10
and abs(FacePose.roll * 180/3.14159) <= 20
and abs(FacePose.pitch * 180/3.14159) <= 20
and Faces.file_id != 'image21068.jpg'
        LIMIT 100""")
results = c.fetchall()


# In[6]:

#print(results[0:5])
#
#
## In[7]:
#
#len(results)


# In[8]:

fname_index = 1 
x_index = 2
y_index = 3
width_index = 4
height_index = 5


# In[9]:

np.array(results[3][x_index:height_index+1],dtype=np.float32).reshape(-1,2)


# In[10]:

def show_bbs(image, bb):
    """show image with bounding box"""
    plt.imshow(image)
    plt.scatter([bb[0], bb[0], bb[0]+bb[2], bb[0]+bb[2]], [bb[1],bb[1]+bb[3],bb[1],bb[1]+bb[3]],marker='.', c='r')
    plt.pause(0.001)


# In[11]:

#i = 4
#plt.figure()
#show_bbs(io.imread(os.path.join(path_to_tmp, results[i][fname_index])),(results[i][x_index],results[i][y_index],results[i][width_index],results[i][height_index]))
#plt.show()


# results = (face_id, face_index, bounding box coords)

# In[62]:

class FaceBBsDataset(Dataset):
    """Face bounding boxes dataset"""
    def __init__(self, results, root_dir, transform_pos=None, transform_neg=None):
        """
        Args:
            results(list of tuples): see above for format
            root_dir(string): dir with all images
        """
        self.image_bbs= results
        self.root_dir = root_dir
        self.transform_pos = transform_pos
        self.transform_neg = transform_neg
        
    def __len__(self):
        return len(self.image_bbs)*2
    
    def __getitem__(self, idx_original):
#        print('idx_original',idx_original)
        if idx_original >= len(self.image_bbs): idx = idx_original - len(self.image_bbs)
        else: idx = idx_original
#        print('idx',idx)
        img_name = os.path.join(self.root_dir, self.image_bbs[idx][fname_index])
        #print('img_name', img_name)
        try:
            image = io.imread(img_name)
            #print(self.image_bbs[idx][x_index:height_index+1])
            bbs = np.array(self.image_bbs[idx][x_index:height_index+1],dtype=np.float32).reshape(-1,2)

            sample = {'image':image, 'bb': bbs}

            if idx_original < len(self.image_bbs):
                if self.transform_pos:
                    sample = self.transform_pos(sample)
            else:
                if self.transform_neg:
                    sample = self.transform_neg(sample)

            return sample
        except: 
            traceback.print_exc()
            blank_image = np.zeros((10,10,3), np.uint8)
            sample = {'image': blank_image, 'bb': np.array((0,0,10,10),dtype=np.float32).reshape(-1,2)}
            if idx_original < len(self.image_bbs):
                if self.transform_pos:
                    sample = self.transform_pos(sample)
            else:
                if self.transform_neg:
                    sample = self.transform_neg(sample)
            return sample


# In[63]:

#frontal_dataset = FaceBBsDataset(results, root_dir=path_to_tmp)
#fig = plt.figure() 
## for i in range(len(frontal_dataset)):
#sample = frontal_dataset[5]
#show_bbs(sample['image'], sample['bb'])



# In[24]:

#fig = plt.figure()
#for i in range(len(frontal_dataset)):
#    sample = frontal_dataset[i]
#    print(i,sample['image'].shape, sample['bb'].shape)
#    
#    ax = plt.subplot(1,4,i+1)
#    plt.tight_layout()
#    ax.set_title('Sample #{}'.format(i))
#    ax.axis('off')
#    show_bbs(**sample)
#    
#    if i == 3:
#        plt.show()
#        break
#

# In[25]:

def iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    #print(boxA, boxB)
    xA = max(boxA[0,0], boxB[0,0])
    xB = min(boxA[0,0]+boxA[1,0], boxB[0,0]+boxB[0,1])
    
    yA = max(boxA[0,1], boxB[0,1])
    yB = min(boxA[0,1]+boxA[1,0], boxB[0,1]+boxB[1,0])
    
    # compute the area of intersection rectangle
    interArea = abs((xB - xA + 1) * (yB - yA + 1))
    #     print(interArea)
    
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = boxA[1,0] * boxA[1,1]
    boxBArea = boxB[1,0] * boxB[1,1]
    #     print(boxAArea)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    # return the intersection over union value
    return iou


# In[ ]:




# In[26]:

class CropFace(object):
    """Crops region containing face in an image in a sample
        Args:
            output_size(tuple or int): desired output size"""
#     def __init__(self):
#         assert isinstance(output_size, (int,tuple))
#         if isinstance(output_size, int):
#             self.output_size = (output_size, output_size)
#         else:
#             assert len(output_size) == 2
#             self.output_size = output_size
    
    def __call__(self, sample):
        image, bb = sample['image'], sample['bb']
        #print("cropface", bb)
        h, w = int(bb[1,1]), int(bb[1,0])
        left, top = int(max(bb[0,0],0)), int(max(bb[0,1],0))
        
        face = image[top: top + h, left: left + w]
        
        return {'x': face, 'y': 1} 
        


# In[27]:

class RandomNonFaceCrop(object):
    """crop a random part of image that is < 0.3 
    IOU with face"""
#     def __init__(self):
    def __call__(self,sample):
        image, bb = sample['image'], sample['bb']
        #print("nonfacecrop", bb)
        #print('bb',bb)
        im_h, im_w = image.shape[:2]
        face_h, face_w = int(bb[1,1]), int(bb[1,0])
        
        if (im_h - face_h > 0) and (im_w - face_w > 0):
            top = np.random.randint(0, im_h - face_h)
            left = np.random.randint(0, im_w - face_w)
            count = 0
            boxB = np.array((top,left,face_w,face_h)).reshape(-1,2)
            comp = iou(bb, boxB)
        else:
            top = 0
            left = 0
            count = 6
            comp = 0.5
            
        while comp > 0.3 and count <= 5:
            top = np.random.randint(0, im_h - face_h)
            left = np.random.randint(0, im_w - face_w)
            boxB = np.array((top,left,face_w,face_h)).reshape(-1,2)
            comp = iou(bb, boxB)
            count += 1
        
        if comp > 0.3 and count > 5:
            image_cropped = image[top: top + face_h, left: left + face_w]
            return {'x': image_cropped, 'y': 1}
        else:
            image_cropped = image[top: top + face_h, left: left + face_w]
            return {'x': image_cropped, 'y': 0}
        


# In[28]:

class Rescale(object):
    """rescale image in sample to a given size
    args:
        output_size (tuple or int)"""
    
    def __init__(self,output_size):
        assert isinstance(output_size, (int,tuple))
        self.output_size = output_size
        
    def __call__(self,sample):
        x, y = sample['x'], sample['y']
        h, w = x.shape[:2]
        
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(x, (new_h, new_w))
        return {'x': img, 'y': y}


# In[ ]:




# In[ ]:




# In[38]:

class ToTensor(object):
    """convert ndarrays in sample to Tensors"""
    def __call__(self,sample):
        x, y = sample['x'], sample['y']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        
        #not face
        if y == 0: y_new = torch.LongTensor([0])
        #face
        if y == 1: y_new = torch.LongTensor([1])
            
        image = x.transpose((2, 0, 1))
        return {'x': torch.from_numpy(image).float(),
                'y': y_new
               }


    
class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B), will normalize each channel of the torch.*Tensor, i.e. channel = (channel - mean) / std"""
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self,sample):
        """tensor (Tensor): Tensor image of size (C, H, W) to be normalized."""
        x,y = sample['x'], sample['y']
        
        for t,m,s in zip(x, self.mean, self.std):
            t.sub_(m).div_(s)
        return {'x': x, 'y': y}
        
    
# In[ ]:




# In[30]:

#crop_face = CropFace()
#crop_nonface = RandomNonFaceCrop()
#scale = Rescale(100)
#composed_f = transforms.Compose([CropFace(), Rescale(100)])
#composed_n = transforms.Compose([RandomNonFaceCrop(), Rescale(100)])
#

# In[189]:

#fig = plt.figure()
#sample = frontal_dataset[2]
#show_bbs(sample['image'], sample['bb'])
#for i, tsfrm in enumerate([crop_face, crop_nonface, composed_f, composed_n]):
##     print(transformed_sample.keys())
##     print(sample.keys())
#    transformed_sample = tsfrm(sample)
##     print(transformed_sample)
##     print(sample)
#    ax = plt.subplot(1,4,i+1)
#    plt.tight_layout()
#    ax.set_title(type(tsfrm).__name__)
#    plt.imshow(transformed_sample['x'])
#plt.show()


# In[64]:

transformed_dataset_face = FaceBBsDataset(results, 
                                     root_dir=path_to_tmp,
                                    transform_pos=transforms.Compose([
                                        CropFace(), 
                                        Rescale((224,224)),
                                        ToTensor()
                                    ]),
                                    transform_neg=transforms.Compose([
                                        RandomNonFaceCrop(), 
                                        Rescale((224,224)),
                                        ToTensor()
                                    ]))
transformed_dataset_nonface = FaceBBsDataset(results, 
                                     root_dir=path_to_tmp,
                                    transform_pos=transforms.Compose([
                                        RandomNonFaceCrop(), 
                                        Rescale((224,224)),
                                        ToTensor()
                                    ]))


# In[65]:

#print(len(transformed_dataset_face))
# print(len(transformed_dataset_nonface))


# In[48]:

#for i in range(20,len(transformed_dataset_face)):
#    try:
#        sample = transformed_dataset_face[i]
#        print(i,sample['x'].size(), sample['y'])
#        if i == 6:
#            break
#    except:
#        print('oops')


# In[30]:

#for i in range(len(transformed_dataset_nonface)):
#    sample = transformed_dataset_nonface[i]
#    print(i,sample['x'].size(), sample['y'])
#    if i == 6:
#        break


# In[68]:

#dataloader = DataLoader(transformed_dataset_face, 
#                        batch_size=4,
#                       shuffle=True, num_workers=2)


# In[72]:

def show_batch(sample_batched):
    """Show image and label for a batch of samples."""
    images_batch, labels_batch = sample_batched['x'], sample_batched['y']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
#         plt.label(labels_batch[i,:])
        plt.text(224*i,0.65,labels_batch[i,:].numpy())


#for i_batch, sample_batched in enumerate(dataloader):
#    try:
#        print(i_batch, sample_batched['x'].size(),
#              sample_batched['y'].size())
#
#        # observe 4th batch and stop.
#        if i_batch == 3:
#            plt.figure()
#            show_batch(sample_batched)
#            plt.axis('off')
#            plt.ioff()
#            plt.show()
#            break
#    except:
#        print('oops')


# In[ ]:




# In[ ]:




# In[ ]:



