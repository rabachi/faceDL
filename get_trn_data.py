import sqlite3
path_to_db = '/Users/romina/Databases/aflw/data/aflw.sqlite'
path_to_imgs = '/Users/romina/Databases/aflw/data/flickr/'
path_to_positives = '/Users/romina/Documents/Masters/pos_training/'
#path_to_tmp = '/Users/romina/Documents/Masters/tmp/'
import os, os.path
import numpy as np
from random import randint
from sklearn.metrics import jaccard_similarity_score
from PIL import Image
from pylab import *
from sklearn.feature_extraction import image
from sklearn.metrics.pairwise import cosine_similarity
import bcolz
from sklearn.utils import shuffle
from shutil import copyfile

def save_array(fname, arr): c=bcolz.carray(arr, rootdir=fname, mode='w'); c.flush()
def load_array(fname): return bcolz.open(fname)[:]

def im_resize(im, sz):
    pil_im = Image.fromarray(uint8(im))
    return array(pil_im.resize(sz))

def random_bbox(im, box_size, box_num):
    width = im.shape[0]
    height = im.shape[1]
    #     print(box_size[0], box_size[1])
    #     print(width, height)
    maxX = width-box_size[0]
    maxY = height-box_size[1]
    
    if (maxX <= 0) or (maxY <= 0):
        return np.empty(0)

    print("maxX, maxY:", maxX, maxY)
    #     print(width-box_size[0], height-box_size[1])
    bboxes = np.array([[randint(0,maxY),
                    randint(0,maxX),
                    box_size[0], box_size[1]]
                   for i in range(box_num)])
    return bboxes

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    
    yA = max(boxA[1], boxB[1])
    yB = min(boxA[1]+boxA[2], boxB[1]+boxB[2])
    
    # compute the area of intersection rectangle
    interArea = abs((xB - xA + 1) * (yB - yA + 1))
    #     print(interArea)
    
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    #     print(boxAArea)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    # return the intersection over union value
    return iou



fname_index = 1
x_index = 2
y_index = 3
width_index = 4
height_index = 5

if __name__ == "__main__":
    
#    all_data1 = load_array(path_to_positives+"all_data.bc")
#    labels1 = load_array(path_to_positives+"labels.bc")
#
#    all_data2 = load_array(path_to_positives+"all_data2.bc")
#    labels2 = load_array(path_to_positives+"labels2.bc")
#
#    all_data3 = load_array(path_to_positives+"all_data3.bc")
#    labels3 = load_array(path_to_positives+"labels3.bc")
#    
#    all_data4 = load_array(path_to_positives+"all_data4.bc")
#    labels4 = load_array(path_to_positives+"labels4.bc")
#
#    all_data5 = load_array(path_to_positives+"all_data5.bc")
#    labels5 = load_array(path_to_positives+"labels5.bc")
#
#
#    X = np.concatenate((all_data1, all_data2, all_data3, all_data4, all_data5),axis=0)
#    y = np.concatenate((labels1, labels2, labels3, labels4, labels5),axis=0)
#    X, y = shuffle(X, y, random_state=0)
#
#    save_array(path_to_positives+"X.bc",X)
#    save_array(path_to_positives+"y.bc",y)
#
#    print(X.shape)
#    print(y.shape)

    all_images = {}
    sub = os.listdir(path_to_imgs)
    for p in sub:
        #     print(p)
        pDir = os.path.join(path_to_imgs, p)
        #     print(pDir)
        if os.path.isdir(pDir):
            this_dir = os.listdir(pDir)
            for i in range(len(this_dir)):
                #             print(os.path.join(p, this_dir[i]))
                all_images[this_dir[i]] = p
    # print(all_images)

    conn = sqlite3.connect(path_to_db)
    c = conn.cursor()
    c.execute("""select Faces.face_id, Faces.file_id,x,y,w,h
from FacePose 
join Faces on FacePose.face_id = Faces.face_id
join FaceRect on FacePose.face_id = FaceRect.face_id
where abs(FacePose.yaw * 180/3.14159) <= 10
and abs(FacePose.roll * 180/3.14159) <= 20
and abs(FacePose.pitch * 180/3.14159) <= 20
        LIMIT 100""")
    results = c.fetchall()
    #for tup in results:
        #fname = tup[fname_index]
        #try:
            #copyfile(os.path.join(path_to_imgs,all_images[fname],fname), os.path.join(path_to_tmp,fname))
        #except Exception as e:
        #    print(e)
#
    pos_count = 0
    neg_count = 0
    all_positives = np.empty((5000,3,224,224), int)
    all_negatives = np.empty((5000,3,224,224), int)
    # print(all_positives)
    for tup in results:
        fname = tup[fname_index]
        print(fname)
        try:
            pil_im0 = np.array(Image.open(os.path.join(path_to_imgs,all_images[fname],fname)))
            if len(pil_im0.shape) == 3 and pil_im0.shape[2] == 3:
                pil_im = pil_im0 - np.mean(pil_im0, axis=(0,1))
                #     imshow(pil_im)
                #     show()
                box_gt = (max(tup[x_index],2), max(tup[y_index],2), tup[width_index], tup[height_index])
                print(box_gt)
                print(pil_im.shape)
                region = pil_im[box_gt[1]:box_gt[1]+box_gt[3],box_gt[0]:box_gt[0]+box_gt[2],:]
                          #     print(region.shape)
                resized = im_resize(region, (224,224))
                all_positives[pos_count] = resized.reshape(3,224,224)
                pos_count +=1
                bboxes = random_bbox(pil_im, (box_gt[2],box_gt[3]), 2)
                for i in range(bboxes.shape[0]):
                    bbox = bboxes[i]
                    patch = pil_im[bbox[1]:bbox[1]+box_gt[3],bbox[0]:bbox[0]+box_gt[2],:]
                    #         print(patch.shape)
                    patch_resized = im_resize(patch, (224,224))
                    overlap = bb_intersection_over_union(bbox, box_gt)
                    #         print("overlap: ", overlap)
                    patch = pil_im[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2],:]
                    #         print(patch.shape)
                    patch_resized = im_resize(patch, (224,224))
                    #         imshow(patch)
                    #         show()
                    if overlap >= 0.5:
                        #             imshow(patch)
                        #             show()
                        all_positives[pos_count] = patch_resized.reshape(3,224,224)
                        pos_count += 1
                    elif overlap <= 0.3:
                        #             imshow(patch)
                        #             show()
                        all_negatives[neg_count] = patch_resized.reshape(3,224,224)
                        neg_count += 1

        except:
            print("oops, couldn't work with ", fname)

    mask = np.all(np.isnan(all_positives) | np.equal(all_positives, 0), axis=3)
    num = int(all_positives[~mask].shape[0]/3/224)
    print(num)
    all_positives = all_positives[~mask].reshape(num,3,224,224)

    mask = np.all(np.isnan(all_negatives) | np.equal(all_negatives, 0), axis=3)
    num = int(all_negatives[~mask].shape[0]/3/224)
    all_negatives = all_negatives[~mask].reshape(num,3,224,224)

    num_labels = all_positives.shape[0] + all_negatives.shape[0]
    labels = np.empty((num_labels, 2), int)

    labels[0:all_positives.shape[0]] = [0,1]
    labels[all_positives.shape[0]:all_positives.shape[0]+all_negatives.shape[0]] = [1,0]

    all_data = np.concatenate((all_positives, all_negatives),axis=0)
    all_data, labels = shuffle(all_data, labels, random_state=0)
#
#    save_array(path_to_positives+"all_data1.bc",all_data)
#    save_array(path_to_positives+"labels1.bc",labels)
#
#
#
