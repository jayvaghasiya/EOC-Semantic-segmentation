import albumentations as A
import os
import cv2
import numpy as np
import os
import cv2
import skimage
from skimage import img_as_ubyte
import numpy as np


def yoloFormattocv(x1, y1, x2, y2, H, W):
    bbox_width = x2 * W
    bbox_height = y2 * H
    center_x = x1 * W
    center_y = y1 * H
    voc = []
    voc.append(center_x - (bbox_width / 2))
    voc.append(center_y - (bbox_height / 2))
    voc.append(center_x + (bbox_width / 2))
    voc.append(center_y + (bbox_height / 2))
    return [int(v) for v in voc]


def cvFormattoYolo(corner, H, W):
    bbox_W = corner[3] - corner[1]
    bbox_H = corner[4] - corner[2]
    center_bbox_x = (corner[1] + corner[3]) / 2
    center_bbox_y = (corner[2] + corner[4]) / 2
    return corner[0], round(center_bbox_x / W, 6), round(center_bbox_y / H, 6), round(bbox_W / W, 6), round(bbox_H / H,
                                                                                                            6)


def findFiles(filename, directory):
    import os
    fullPath = ''
    for root, dir, files in os.walk(directory):
        if filename in files:
            fullPath = os.path.join(root, filename)
    return fullPath


def flip_horizontal(img, imageName, augmentedPath, annotationFilePath, width, height, showImages):
    # create and save the new image
    newImage = cv2.flip(img, 1)
    mask = cv2.imread(annotationFilePath)
    newMask = cv2.flip(mask,1)
    newImageName = imageName + '_horizontalFlip.jpg'
    newImagePath = os.path.join(augmentedPath, newImageName)
    cv2.imwrite(newImagePath, newImage)

    
    newAnnotationName = imageName + '_horizontalFlip.png'
    newAnnotationPath = os.path.join(augmentedPath, newAnnotationName)
    cv2.imwrite(newAnnotationPath,newMask)


    if showImages:
        cv2.imshow('flipHorizontal', newImage)
    return newImagePath, newAnnotationPath


def flip_vertical(img, imageName, augmentedPath, annotationFilePath, width, height, showImages):
    # create and save the new image
    newImage = cv2.flip(img, 0)
    mask = cv2.imread(annotationFilePath)
    newMask = cv2.flip(mask,0)
    newImageName = imageName + '_verticalFlip.jpg'
    newImagePath = os.path.join(augmentedPath, newImageName)
    cv2.imwrite(newImagePath, newImage)

    
    newAnnotationName = imageName + '_verticalFlip.png'
    newAnnotationPath = os.path.join(augmentedPath, newAnnotationName)
    cv2.imwrite(newAnnotationPath,newMask)

    if showImages:
        cv2.imshow('flipVertical', newImage)


def flip_full(img, imageName, augmentedPath, annotationFilePath, width, height, showImages):
    # create and save the new image
    newImage = cv2.flip(img, -1)
    mask = cv2.imread(annotationFilePath)
    newMask = cv2.flip(mask,-1)
    newImageName = imageName + '_fullFlip.jpg'
    newImagePath = os.path.join(augmentedPath, newImageName)
    cv2.imwrite(newImagePath, newImage)

    # create and save the new annotation file
   
    newAnnotationName = imageName + '_fullFlip.png'
    newAnnotationPath = os.path.join(augmentedPath, newAnnotationName)
    cv2.imwrite(newAnnotationPath,newMask)


    if showImages:
        cv2.imshow('flipFull', newImage)







# In[7]:


def saltPepper(img, imageName, augmentedPath, annotationFilePath, width, height, showImages):
    # create and save the new image
    newImage = skimage.util.random_noise(img, mode='s&p', seed=None, clip=True, amount=0.05)
    newImage = img_as_ubyte(newImage)
    newImageName = imageName + '_saltPepper.jpg'
    newImagePath = os.path.join(augmentedPath, newImageName)
    cv2.imwrite(newImagePath, newImage)
    newAnnotationName = imageName + '_saltPepper.png'
    newAnnotationPath = os.path.join(augmentedPath, newAnnotationName)
    shutil.copy(annotationFilePath, newAnnotationPath)
    
    if showImages:
        cv2.imshow('saltPepper', newImage)


# In[8]:


def gaussian(img, imageName, augmentedPath, annotationFilePath, width, height, showImages):
    import os
    import cv2
    import skimage
    from skimage import img_as_ubyte
    # import numpy as np
    import random

    # create and save the new image
    randomMean = (random.random() / 2) - 0.25
    newImage = skimage.util.random_noise(img, mode='gaussian', seed=None, clip=True, mean=randomMean)
    newImage = img_as_ubyte(newImage)
    newImageName = imageName + '_gaussian.jpg'
    newImagePath = os.path.join(augmentedPath, newImageName)
    cv2.imwrite(newImagePath, newImage)
    # create and save the new annotation file

    newAnnotationName = imageName + '_gaussian.png'
    newAnnotationPath = os.path.join(augmentedPath, newAnnotationName)
    shutil.copy(annotationFilePath, newAnnotationPath)

    if showImages:
        cv2.imshow('gaussian', newImage)
    return newImagePath, newAnnotationPath


# In[9]:



# In[10]:


import os
import cv2
import shutil




blur_limit = (3, 19)


def AdvancedBlur(img, imageName, augmentedPath, annotationFilePath, width, height, showImages):
    import albumentations as A
    # Declare an augmentation pipeline
    transform = A.Compose([
        A.AdvancedBlur(blur_limit, p=1),
    ])
    # Augment an image
    transformed = transform(image=img)
    transformed_image = transformed["image"]
    newImageName = imageName + '_advancedblur.jpg'
    newImagePath = os.path.join(augmentedPath, newImageName)
    cv2.imwrite(newImagePath, transformed_image)

    newAnnotationName = imageName + '_advancedblur.png'
    newAnnotationPath = os.path.join(augmentedPath, newAnnotationName)
    shutil.copy(annotationFilePath, newAnnotationPath)

    return newImagePath, newAnnotationPath


# In[63]:


def MedianBlur(img, imageName, augmentedPath, annotationFilePath, width, height, showImages):

    # Declare an augmentation pipeline
    transform = A.Compose([
        A.MedianBlur(blur_limit),
    ])
    # Augment an image
    transformed = transform(image=img)
    transformed_image = transformed["image"]
    newImageName = imageName + '_medianblur.jpg'
    newImagePath = os.path.join(augmentedPath, newImageName)
    cv2.imwrite(newImagePath, transformed_image)
    newAnnotationName = imageName + '_medianblur.png'
    newAnnotationPath = os.path.join(augmentedPath, newAnnotationName)
    shutil.copy(annotationFilePath, newAnnotationPath)

    return newImagePath, newAnnotationPath


# In[64]:


def Defocus(img, imageName, augmentedPath, annotationFilePath, width, height, showImages):
    # Declare an augmentation pipeline
    transform = A.Compose([
        A.Defocus(blur_limit),
    ])
    # Augment an image
    transformed = transform(image=img)
    transformed_image = transformed["image"]
    newImageName = imageName + '_defocus.jpg'
    newImagePath = os.path.join(augmentedPath, newImageName)
    cv2.imwrite(newImagePath, transformed_image)
    newAnnotationName = imageName + '_defocus.png'
    newAnnotationPath = os.path.join(augmentedPath, newAnnotationName)
    shutil.copy(annotationFilePath, newAnnotationPath)

    return newImagePath, newAnnotationPath


# In[65]:


def MotionBlur(img, imageName, augmentedPath, annotationFilePath, width, height, showImages):
    # Declare an augmentation pipeline
    transform = A.Compose([
        A.MotionBlur(blur_limit, p=1),
    ])
    # Augment an image
    transformed = transform(image=img)
    transformed_image = transformed["image"]
    newImageName = imageName + '_motionblur.jpg'
    newImagePath = os.path.join(augmentedPath, newImageName)
    cv2.imwrite(newImagePath, transformed_image)
    newAnnotationName = imageName + '_motionblur.png'
    newAnnotationPath = os.path.join(augmentedPath, newAnnotationName)
    shutil.copy(annotationFilePath, newAnnotationPath)

    return newImagePath, newAnnotationPath


# In[66]:


def ZoomBlur(img, imageName, augmentedPath, annotationFilePath, width, height, showImages):
    # Declare an augmentation pipeline
    transform = A.Compose([
        A.ZoomBlur(blur_limit),
    ])
    # Augment an image
    transformed = transform(image=img)
    transformed_image = transformed["image"]
    newImageName = imageName + '_zoomblur.jpg'
    newImagePath = os.path.join(augmentedPath, newImageName)
    cv2.imwrite(newImagePath, transformed_image)
    newAnnotationName = imageName + '_zoomblur.png'
    newAnnotationPath = os.path.join(augmentedPath, newAnnotationName)
    shutil.copy(annotationFilePath, newAnnotationPath)

    return newImagePath, newAnnotationPath


# In[67]:


def GlassBlur(img, imageName, augmentedPath, annotationFilePath, width, height, showImages):

    # Declare an augmentation pipeline
    transform = A.Compose([
        A.GlassBlur(),
    ])
    # Augment an image
    transformed = transform(image=img)
    transformed_image = transformed["image"]
    newImageName = imageName + '_glassblur.jpg'
    newImagePath = os.path.join(augmentedPath, newImageName)
    cv2.imwrite(newImagePath, transformed_image)
    newAnnotationName = imageName + '_glassblur.png'
    newAnnotationPath = os.path.join(augmentedPath, newAnnotationName)
    shutil.copy(annotationFilePath, newAnnotationPath)

    return newImagePath, newAnnotationPath


# In[68]:


def flipScaleBrightness(img, imageName, augmentedPath, annotationFilePath, width, height, showImages):

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3),
    ],
    )
    mask = cv2.imread(annotationFilePath)
    transformed = transform(image=img, mask=mask)
    transformed_image = transformed["image"]
    transformed_mask = transformed["mask"]
    newImageName = imageName + '_flipscale.jpg'
    newAnnotationName = imageName + '_flipscale.png'
    newImagePath = os.path.join(augmentedPath, newImageName)
    newAnnotationPath = os.path.join(augmentedPath, newAnnotationName)
    cv2.imwrite(newImagePath, transformed_image)
    cv2.imwrite(newAnnotationPath, transformed_mask)
    
    return newImagePath, newAnnotationPath


# In[69]:


def RandomCrop(img, imageName, augmentedPath, annotationFilePath, width, height, showImages):

    transform = A.Compose([
        A.RandomCrop(width=640, height=640),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])
    mask = cv2.imread(annotationFilePath)
    transformed = transform(image=img, mask=mask)
    transformed_image = transformed["image"]
    transformed_mask = transformed["mask"]
    newImageName = imageName + '_randomcrop.jpg'
    newAnnotationName = imageName + '_randomcrop.png'
    newImagePath = os.path.join(augmentedPath, newImageName)
    newAnnotationPath = os.path.join(augmentedPath, newAnnotationName)
    cv2.imwrite(newImagePath, transformed_image)
    cv2.imwrite(newAnnotationPath, transformed_mask)
    
    return newImagePath, newAnnotationPath


# In[70]:


def CenterCrop(img, imageName, augmentedPath, annotationFilePath, width, height, showImages):

    transform = A.Compose(
        [A.CenterCrop(height=640, width=640, p=1)],
    )
    mask = cv2.imread(annotationFilePath)
    transformed = transform(image=img,mask=mask)
    transformed_image = transformed["image"]
    transformed_mask = transformed["mask"]
    newImageName = imageName + '_centercrop.jpg'
    newAnnotationName = imageName + '_centercrop.png'
    newImagePath = os.path.join(augmentedPath, newImageName)
    newAnnotationPath = os.path.join(augmentedPath, newAnnotationName)
    cv2.imwrite(newImagePath, transformed_image)
    cv2.imwrite(newAnnotationPath, transformed_mask)
   
    return newImagePath, newAnnotationPath


# In[71]:


def customaug(img, imageName, augmentedPath, annotationFilePath, width, height, showImages):

    transform = A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.1),
            A.IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.IAASharpen(),
            A.IAAEmboss(),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
    ], p=0.5)

    mask = cv2.imread(annotationFilePath)
    transformed = transform(image=img, mask=mask)
    transformed_image = transformed["image"]
    transformed_mask = transformed["mask"]
    newImageName = imageName + '_customaug.jpg'
    newAnnotationName = imageName + '_customaug.png'
    newImagePath = os.path.join(augmentedPath, newImageName)
    newAnnotationPath = os.path.join(augmentedPath, newAnnotationName)
    cv2.imwrite(newImagePath, transformed_image)
    cv2.imwrite(newAnnotationPath, transformed_mask)
    
    return newImagePath, newAnnotationPath


# In[72]:


def customaug_l(img, imageName, augmentedPath, annotationFilePath, width, height, showImages):

    transform = A.Compose([
        A.HorizontalFlip(p=1),
        A.RandomSizedCrop((600 - 100, 600 + 100), 600, 600),
        A.GaussNoise(var_limit=(100, 150), p=1),
    ],  p=0.5)

    mask = cv2.imread(annotationFilePath)
    transformed = transform(image=img, mask=mask)
    transformed_image = transformed["image"]
    transformed_mask = transformed["mask"]
    newImageName = imageName + '_customaugl.jpg'
    newAnnotationName = imageName + '_customaugl.png'
    newImagePath = os.path.join(augmentedPath, newImageName)
    newAnnotationPath = os.path.join(augmentedPath, newAnnotationName)
    cv2.imwrite(newImagePath, transformed_image)
    cv2.imwrite(newAnnotationPath, transformed_mask)

    return newImagePath, newAnnotationPath


# In[73]:


def customaug_m(img, imageName, augmentedPath, annotationFilePath, width, height, showImages):

    transform = A.Compose([
        A.HorizontalFlip(p=1),
        A.RandomSizedCrop((600 - 100, 600 + 100), 600, 600),
        A.MotionBlur(blur_limit=17, p=1),
    ], p=0.5)
    
    mask = cv2.imread(annotationFilePath)
    transformed = transform(image=img, mask=mask)
    transformed_image = transformed["image"]
    transformed_mask = transformed["mask"]
    newImageName = imageName + '_customaugm.jpg'
    newAnnotationName = imageName + '_customaugm.png'
    newImagePath = os.path.join(augmentedPath, newImageName)
    newAnnotationPath = os.path.join(augmentedPath, newAnnotationName)
    cv2.imwrite(newImagePath, transformed_image)
    cv2.imwrite(newAnnotationPath, transformed_mask)

    return newImagePath, newAnnotationPath


# In[74]:


def customaug_s(img, imageName, augmentedPath, annotationFilePath, width, height, showImages):

    transform = A.Compose([
        A.HorizontalFlip(p=1),
        A.RandomSizedCrop((600 - 100, 600 + 100), 600, 600),
        A.RGBShift(p=1),
        A.Blur(blur_limit=11, p=1),
        A.RandomBrightnessContrast(p=1),
        A.CLAHE(p=1),
    ], p=0.5)

    # create and save the new annotation file
    
    mask = cv2.imread(annotationFilePath)
    transformed = transform(image=img, mask=mask)
    transformed_image = transformed["image"]
    transformed_mask = transformed["mask"]
    newImageName = imageName + '_customaugs.jpg'
    newAnnotationName = imageName + '_customaugs.png'
    newImagePath = os.path.join(augmentedPath, newImageName)
    newAnnotationPath = os.path.join(augmentedPath, newAnnotationName)
    cv2.imwrite(newImagePath, transformed_image)
    cv2.imwrite(newAnnotationPath, transformed_mask)
    return newImagePath, newAnnotationPath


# In[ ]:


def RandomFog(img, imageName, augmentedPath, annotationFilePath, width, height, showImages):

    # Declare an augmentation pipeline
    transform = A.Compose([
        A.RandomFog(),
    ])
    # Augment an image
    transformed = transform(image=img)
    transformed_image = transformed["image"]
    newImageName = imageName + '_fog.jpg'
    newImagePath = os.path.join(augmentedPath, newImageName)
    cv2.imwrite(newImagePath, transformed_image)
    newAnnotationName = imageName + '_fog.png'
    newAnnotationPath = os.path.join(augmentedPath, newAnnotationName)
    shutil.copy(newAnnotationPath, newAnnotationPath)

    return newImagePath, newAnnotationPath


# In[97]:


def RandomShadow(img, imageName, augmentedPath, annotationFilePath, width, height, showImages):

    # Declare an augmentation pipeline
    transform = A.Compose([
        A.RandomShadow(num_shadows_lower=3, num_shadows_upper=5, shadow_dimension=7, always_apply=True),
    ])
    # Augment an image
    transformed = transform(image=img)
    transformed_image = transformed["image"]
    newImageName = imageName + '_shadow.jpg'
    newImagePath = os.path.join(augmentedPath, newImageName)
    cv2.imwrite(newImagePath, transformed_image)
    newAnnotationName = imageName + '_shadow.png'
    newAnnotationPath = os.path.join(augmentedPath, newAnnotationName)
    shutil.copy(newAnnotationPath, newAnnotationPath)

    return newImagePath, newAnnotationPath


# In[101]:


def RandomSunFlare(img, imageName, augmentedPath, annotationFilePath, width, height, showImages):
    # Declare an augmentation pipeline
    transform = A.Compose([
        A.RandomSunFlare(),
    ])
    # Augment an image
    transformed = transform(image=img)
    transformed_image = transformed["image"]
    newImageName = imageName + '_sunflare.jpg'
    newImagePath = os.path.join(augmentedPath, newImageName)
    cv2.imwrite(newImagePath, transformed_image)
    newAnnotationName = imageName + '_sunflare.png'
    newAnnotationPath = os.path.join(augmentedPath, newAnnotationName)
    shutil.copy(newAnnotationPath, newAnnotationPath)
    return newImagePath, newAnnotationPath


# In[112]:


def Solarize(img, imageName, augmentedPath, annotationFilePath, width, height, showImages):

    # Declare an augmentation pipeline
    transform = A.Compose([
        A.Solarize(p=1),
    ])
    # Augment an image
    transformed = transform(image=img)
    transformed_image = transformed["image"]
    newImageName = imageName + '_solarize.jpg'
    newImagePath = os.path.join(augmentedPath, newImageName)
    cv2.imwrite(newImagePath, transformed_image)
    newAnnotationName = imageName + '_solarize.png'
    newAnnotationPath = os.path.join(augmentedPath, newAnnotationName)
    shutil.copy(newAnnotationPath, newAnnotationPath)

    return newImagePath, newAnnotationPath


# In[115]:


def Spatter(img, imageName, augmentedPath, annotationFilePath, width, height, showImages):

    # Declare an augmentation pipeline
    transform = A.Compose([
        A.Spatter(p=1),
    ])
    # Augment an image
    transformed = transform(image=img)
    transformed_image = transformed["image"]
    newImageName = imageName + '_spatter.jpg'
    newImagePath = os.path.join(augmentedPath, newImageName)
    cv2.imwrite(newImagePath, transformed_image)
    newAnnotationName = imageName + '_spatter.png'
    newAnnotationPath = os.path.join(augmentedPath, newAnnotationName)
    shutil.copy(newAnnotationPath, newAnnotationPath)

    return newImagePath, newAnnotationPath


# In[118]:


def UnsharpMask(img, imageName, augmentedPath, annotationFilePath, width, height, showImages):

    # Declare an augmentation pipeline
    transform = A.Compose([
        A.UnsharpMask(p=1),
    ])
    # Augment an image
    transformed = transform(image=img)
    transformed_image = transformed["image"]
    newImageName = imageName + '_UnsharpMask.jpg'
    newImagePath = os.path.join(augmentedPath, newImageName)
    cv2.imwrite(newImagePath, transformed_image)
    newAnnotationName = imageName + '_UnsharpMask.png'
    newAnnotationPath = os.path.join(augmentedPath, newAnnotationName)
    shutil.copy(newAnnotationPath, newAnnotationPath)

    return newImagePath, newAnnotationPath


# In[120]:

if __name__ == "__main__":
    import cv2
    from matplotlib import pyplot as plt

    img_path = "/home/jay/augmentation/notebook/samples/18.png"
    annotation_path = "/home/jay/augmentation/notebook/samples/18.txt"

    in_img = cv2.imread(img_path)

    imageName = img_path.split('/')[-1].split(".")[0]
    augmentedPath = "/home/jay/augmentation/notebook/augmented/"

    height, width = in_img.shape[:2]
    # img_path, ann_path = flip_horizontal(in_img, imageName, augmentedPath, annotation_path, width, height, False)
    # img_path, ann_path = gaussian(in_img, imageName, augmentedPath, annotation_path, width, height, False)

    # img_path, ann_path = AdvancedBlur(in_img, imageName, augmentedPath, annotation_path, width, height, False) # Tested
    # img_path, ann_path = Defocus(in_img, imageName, augmentedPath, annotation_path, width, height, False) # Tested
    # img_path, ann_path = MedianBlur(in_img, imageName, augmentedPath, annotation_path, width, height, False) # Tested
    # img_path, ann_path = MotionBlur(in_img, imageName, augmentedPath, annotation_path, width, height, False) # Tested
    # img_path, ann_path = ZoomBlur(in_img, imageName, augmentedPath, annotation_path, width, height, False) # Tested
    # img_path, ann_path = GlassBlur(in_img, imageName, augmentedPath, annotation_path, width, height, False) # Tested

    # img_path, ann_path = RandomCrop(in_img, imageName, augmentedPath, annotation_path, width, height, False) # Tested
    # img_path, ann_path = flipScaleBrightness(in_img, imageName, augmentedPath, annotation_path, width, height, False) # Tested
    # img_path, ann_path = CenterCrop(in_img, imageName, augmentedPath, annotation_path, width, height, False) # Tested
    # img_path, ann_path = customaug(in_img, imageName, augmentedPath, annotation_path, width, height, False) # Tested

    # img_path, ann_path = customaug_l(in_img, imageName, augmentedPath, annotation_path, width, height, False) # Tested
    # img_path, ann_path = customaug_m(in_img, imageName, augmentedPath, annotation_path, width, height, False) # Tested
    # img_path, ann_path = customaug_s(in_img, imageName, augmentedPath, annotation_path, width, height, False) # Tested

    # img_path, ann_path = RandomFog(in_img, imageName, augmentedPath, annotation_path, width, height, False) # Tested
    # img_path, ann_path = RandomShadow(in_img, imageName, augmentedPath, annotation_path, width, height, False) # Tested
    # img_path, ann_path = RandomSunFlare(in_img, imageName, augmentedPath, annotation_path, width, height, False) # Tested

    # img_path, ann_path = Solarize(in_img, imageName, augmentedPath, annotation_path, width, height, False) # Tested
    # img_path, ann_path = Spatter(in_img, imageName, augmentedPath, annotation_path, width, height, False) # Tested

    output = show_img(img_path, annotation_path)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    plt.imshow(output)
    plt.title('Augmented')
    plt.show()

    # In[ ]:
