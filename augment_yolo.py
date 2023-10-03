from augment_images import *
import os
import cv2
import shutil
import argparse
parser = argparse.ArgumentParser(description='Augmentation Code')
parser.add_argument('--source',
                    type=str,
                        help='path/to/dataset-dir')
parser.add_argument('--destination',
                    type=str,
                        help='path for the augmented dataset')

args = parser.parse_args()

input_dataset_path = args.source
augmentedPath = args.destination
if not os.path.exists(augmentedPath):
    os.makedirs(augmentedPath)
files = os.listdir(input_dataset_path)

image_list = []
lable_list = []

for index, file in enumerate(files):
	if file.endswith(".jpg"):
		# # print(index//2, "/", len(files)//2)
		aug_count = 0
		# image_list.append(os.path.join(input_dataset_path, file))
		# lable_list.append(os.path.join(input_dataset_path, file.replace('jpg', 'txt')))
		# img_path = image_list[-1]
		# annotation_path = lable_list[-1]

		img_path = os.path.join(input_dataset_path, file)
		annotation_path = os.path.join(input_dataset_path, file.replace('jpg', 'png'))
		# gt_img = cv2.imread(img_path)
		# # gt_img = cv2.resize(gt_img,(640,640))
		# gt_label = cv2.imread(annotation_path)
		# # gt_label = cv2.resize(gt_label,(640,640))
		# cv2.imwrite(f'{augmentedPath}/{file}',gt_img)
		# cv2.imwrite(f'{augmentedPath}/{file.replace("jpg", "png")}',gt_label)
		shutil.copy(f'{img_path}',f'{augmentedPath}/{file}')
		shutil.copy(f'{annotation_path}',f'{augmentedPath}/{file.replace("jpg", "png")}')
		print(img_path.split('/')[-1].split(".")[0], index)
		in_img = cv2.imread(img_path)
		imageName = img_path.split('/')[-1].split(".")[0] + "_{}".format(aug_count)
		height, width = in_img.shape[:2]

		for i in range(1):
			try:
				aug_count += 1
				imageName = img_path.split('/')[-1].split(".")[0] + "_{}".format(aug_count)
				flip_horizontal(in_img, imageName, augmentedPath, annotation_path, width, height, False)
				# print(1)
				aug_count += 1
				imageName = img_path.split('/')[-1].split(".")[0] + "_{}".format(aug_count)
				gaussian(in_img, imageName, augmentedPath, annotation_path, width, height, False)
				# # print(2)
				# aug_count += 1
				# imageName = img_path.split('/')[-1].split(".")[0] + "_{}".format(aug_count)
				# AdvancedBlur(in_img, imageName, augmentedPath, annotation_path, width, height, False) # Tested
				# print(3)
				aug_count += 1
				imageName = img_path.split('/')[-1].split(".")[0] + "_{}".format(aug_count)
				# print(4)
				# Defocus(in_img, imageName, augmentedPath, annotation_path, width, height, False) # Tested
				# aug_count += 1
				# imageName = img_path.split('/')[-1].split(".")[0] + "_{}".format(aug_count)
				# # print(5)
				# MedianBlur(in_img, imageName, augmentedPath, annotation_path, width, height, False) # Tested
				# aug_count += 1
				# imageName = img_path.split('/')[-1].split(".")[0] + "_{}".format(aug_count)
				# # print(1)
				# MotionBlur(in_img, imageName, augmentedPath, annotation_path, width, height, False) # Tested
				# # aug_count += 1
				# # imageName = img_path.split('/')[-1].split(".")[0] + "_{}".format(aug_count)
				# # ZoomBlur(in_img, imageName, augmentedPath, annotation_path, width, height, False) # Tested
				# aug_count += 1
				# imageName = img_path.split('/')[-1].split(".")[0] + "_{}".format(aug_count)
				# # print(6)
				# GlassBlur(in_img, imageName, augmentedPath, annotation_path, width, height, False) # Tested

				for i in range(20):
					try:
						aug_count += 1
						imageName = img_path.split('/')[-1].split(".")[0] + "_{}".format(aug_count)
						# print(7, i)
						RandomCrop(in_img, imageName, augmentedPath, annotation_path, width, height, False) # Tested
						aug_count += 1
						imageName = img_path.split('/')[-1].split(".")[0] + "_{}".format(aug_count)
						# print(12)
						customaug_m(in_img, imageName, augmentedPath, annotation_path, width, height, False) # Tested
						aug_count += 1
						imageName = img_path.split('/')[-1].split(".")[0] + "_{}".format(aug_count)
						# print(13)
						customaug_s(in_img, imageName, augmentedPath, annotation_path, width, height, False) # Tested
						aug_count += 1
						imageName = img_path.split('/')[-1].split(".")[0] + "_{}".format(aug_count)
						# print(11)
						customaug_l(in_img, imageName, augmentedPath, annotation_path, width, height, False) # Tested
					except Exception as e:
						print(e)
						continue
				# aug_count += 1
				# imageName = img_path.split('/')[-1].split(".")[0] + "_{}".format(aug_count)
				# # print(8)
				# flipScaleBrightness(in_img, imageName, augmentedPath, annotation_path, width, height, False) # Tested
				# aug_count += 1
				# imageName = img_path.split('/')[-1].split(".")[0] + "_{}".format(aug_count)
				# # print(9)
				# CenterCrop(in_img, imageName, augmentedPath, annotation_path, width, height, False) # Tested
				# aug_count += 1
				# imageName = img_path.split('/')[-1].split(".")[0] + "_{}".format(aug_count)
				# # print(10)
				# customaug(in_img, imageName, augmentedPath, annotation_path, width, height, False) # Tested


				# aug_count += 1
				# imageName = img_path.split('/')[-1].split(".")[0] + "_{}".format(aug_count)
				# RandomFog(in_img, imageName, augmentedPath, annotation_path, width, height, False) # Tested
				# aug_count += 1
				# imageName = img_path.split('/')[-1].split(".")[0] + "_{}".format(aug_count)
				# print(14)
				# RandomShadow(in_img, imageName, augmentedPath, annotation_path, width, height, False) # Tested
				# aug_count += 1
				# imageName = img_path.split('/')[-1].split(".")[0] + "_{}".format(aug_count)
				# RandomSunFlare(in_img, imageName, augmentedPath, annotation_path, width, height, False) # Tested

				# aug_count += 1
				# imageName = img_path.split('/')[-1].split(".")[0] + "_{}".format(aug_count)
				# Solarize(in_img, imageName, augmentedPath, annotation_path, width, height, False) # Tested
				# aug_count += 1
				# imageName = img_path.split('/')[-1].split(".")[0] + "_{}".format(aug_count)
				# Spatter(in_img, imageName, augmentedPath, annotation_path, width, height, False) # Tested

			except Exception as e:
				print("***************",e)
				continue
