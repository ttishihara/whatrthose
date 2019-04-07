import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os 
import pandas as pd
import sys

def grab_paths(data_path):
    imgs = []
    labels = [] 
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if 'label' in file:
                labels.append(os.path.join(root,file))
                continue
            else:
                if file.split('.')[-1]=='npy':
                    imgs.append(os.path.join(root,file))
    imgs.sort()            
    labels.sort()
    return imgs, labels

def plot1(img, label):
    fig, axes = plt.subplots(1,1,figsize=(30,30))
    box = label[0]
    x1 = box[1]
    y1 = box[0]
    x2 = box[3]
    y2 = box[2]
    p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1,
                    alpha=0.7, linestyle="-",
                    edgecolor='red', facecolor='none')
    axes.add_patch(p)
    axes.imshow(img, cmap='gray')

def plot4(imgs,labels):
    idxs = np.random.choice(len(imgs), 4, replace=False)
    fig, axes = plt.subplots(2,2,figsize=(30,30))
    axes = axes.flatten()
    for i in range(len(idxs)):
        box = np.load(labels[idxs[i]])[0]
        x1 = box[1]
        y1 = box[0]
        x2 = box[3]
        y2 = box[2]
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1,
                        alpha=0.7, linestyle="-",
                        edgecolor='red', facecolor='none')
        axes[i].set_title(idxs[i])
        axes[i].add_patch(p)
        axes[i].imshow(np.load(imgs[idxs[i]]),cmap='gray')
        
def rotate(imgs, labels, angles, save_path):
    '''
    Rotate images 90, 180, or 270 degrees and saves to save path
    Inputs:
        - imgs: list of numpy array paths for images 
        - labels : list of numpy array paths for labels 
        - angle: degree of rotation for all images
    Returns:
        - rotated_imgs: list of paths to rotated images 
        - rotated_labels: list of paths to rotated bboxes 
    '''
    
    rotated_imgs = [] 
    rotated_labels = []
    for angle in angles:
        assert angle in [90,180,270], "Invalid angle rotation"
        for idx in range(len(imgs)):
            img = np.load(imgs[idx])
            rows, cols = img.shape[:2]
            # flip the original image 
            if angle == 90:
                img = np.rot90(img,k=1)
            elif angle == 180:
                img = np.rot90(img,k=2)
            elif angle == 270:
                img = np.rot90(img,k=-1)
            # rotate the bounding box coordinates
            rotated_bboxes = np.empty((1,4))
            bboxes = np.load(labels[idx])
            for bbox in bboxes:
                x1_o = bbox[1]
                y1_o = bbox[0]
                x2_o = bbox[3]
                y2_o = bbox[2]
                if angle == 270:
                    x1 = cols - y2_o
                    x2 = cols - y1_o
                    y1 = x1_o
                    y2 = x2_o
                elif angle == 180:
                    x2 = cols - x1_o
                    x1 = cols - x2_o
                    y2 = rows - y1_o
                    y1 = rows - y2_o
                elif angle == 90:
                    x1 = y1_o
                    x2 = y2_o
                    y1 = rows - x2_o
                    y2 = rows - x1_o
                # append rotated bboxes 
                rotated_bboxes = np.append(rotated_bboxes, np.array([[y1,x1,y2,x2]]), axis=0)
            # save rotated bbox to path   
            rotated_label_path = os.path.join(save_path,labels[idx].split('/')[-1].split('.')[0])+'_'+str(angle)+'.npy'
            rotated_labels.append(rotated_label_path)
            np.save(rotated_label_path, rotated_bboxes[1:][:])        
            # save rotated image      
            rotated_img_path = os.path.join(save_path,imgs[idx].split('/')[-1].split('.')[0])+'_'+str(angle)+'.npy'
            rotated_imgs.append(rotated_img_path)
            np.save(rotated_img_path, img)
    return rotated_imgs, rotated_labels 

def vflip(imgs, labels, save_path):
    '''
    Flip images vertically 
    Inputs:
        - imgs: list of numpy array paths for images 
        - labels : list of numpy array paths for labels 
    Returns:
        - imgs: list of paths to flipped images 
        - labels: list of paths to flipped bboxes 
    '''
    
    flipped_imgs = [] 
    flipped_labels = []
    
    for idx in range(len(imgs)):
        img = np.load(imgs[idx])
        img = np.flip(img,axis=1)
        rows, cols = img.shape[:2]
        # flip the bounding box coordinates
        flipped_bboxes = np.empty((1,4))
        bboxes = np.load(labels[idx])
        for bbox in bboxes:        
            y1 = bbox[0]
            y2 = bbox[2]            
            x1_o = bbox[1]
            x2_o = bbox[3]
            x1 = cols - x2_o
            x2 = cols - x1_o
            flipped_bboxes = np.append(flipped_bboxes, np.array([[y1,x1,y2,x2]]), axis=0)
        # save flipped bbox to path   
        flipped_label_path = os.path.join(save_path,labels[idx].split('/')[-1].split('.')[0])+'_v.npy'
        flipped_labels.append(flipped_label_path)
        np.save(flipped_label_path, flipped_bboxes[1:][:])        
        # save flipped image      
        flipped_img_path = os.path.join(save_path,imgs[idx].split('/')[-1].split('.')[0])+'_v.npy'
        flipped_imgs.append(flipped_img_path)
        np.save(flipped_img_path, img)

    return flipped_imgs, flipped_labels 












