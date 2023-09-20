# -*- coding: utf-8 -*-

# Cell 1: Creating the data save

import os
import shutil
from PIL import Image
import pickle
import numpy as np
import cv2
from scipy.io import loadmat
import matplotlib.pyplot as plt
import re

##############################################################################
PICTURE_SIZE = 204
PATCH_SIZE = 180
CROPPED_HEIGHT = 4071
CROPPED_WIDTH = 4964
NUM_PATCHES_HEIGHT = CROPPED_HEIGHT // PATCH_SIZE
NUM_PATCHES_WIDTH = CROPPED_WIDTH // PATCH_SIZE

PATCH_ID = 0
TEST_LEG_SIZE = 7
TRAIN_LEG_SIZE = 100

TEST_THRESHOLD = 99
TRAIN_THRESHOLD = 95

# Write in here the relevent paths for your computer
patches_directory_path = "./data/patches/"
#patches_directory_path = r"C:\Users\lenovo\Desktop\DL_data\patches/"

path_to_mat = r'C:\Users\holtz\OneDrive\Desktop\DL Project by 08.08\5_DataDarkLines'
#path_to_mat = r'C:\Users\lenovo\Desktop\DL_data\5_DataDarkLines'
path_to_img = r'C:\Users\holtz\OneDrive\Desktop\DL Project by 08.08\3_ImagesLinesRemovedBW'
#path_to_img = r'C:\Users\lenovo\Desktop\DL_data\3_ImagesLinesRemovedBW'

##############################################################################

def get_files_in_folder(folder_path):
    """
    Get the names of all files in the specified folder.

    Parameters:
        folder_path (str): The path to the folder.

    Returns:
        list: A list containing the names of all files in the folder.
    """
    file_names = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            file_names.append(file_name)
    return file_names

##############################################################################

def is_image_mostly_black(image, threshold):
    """
    Check if an image is mostly black.

    Parameters:
    image (numpy.ndarray): The image to check, in grayscale.
    threshold (float): The percentage of black pixels above which the image is considered mostly black.

    Returns:
    bool: True if the image is mostly black, False otherwise.
    """
    # Ensure image is in grayscale
    if len(image.shape) > 2:
        raise ValueError("Input image should be grayscale")

    num_black_pixels = np.sum(image == 0)
    total_pixels = image.shape[0] * image.shape[1]
    black_percentage = num_black_pixels / total_pixels * 100

    return black_percentage >= threshold

##############################################################################
def preprocessing_image(img):
    """
    Preprocesses an input image to enhance features and prepare it for further analysis.

    This function applies a series of image processing steps to enhance the features of the input image.
    It employs a combination of thresholding, erosion, and dilation operations to highlight important
    structures and reduce noise in the image.

    Parameters:
    img (numpy.ndarray): The input image to be preprocessed.

    Returns:
    numpy.ndarray: The preprocessed image with enhanced features.

    Note: The input image should be in grayscale (single channel) format.
    """
    _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    binary_img = cv2.erode(binary_img, kernel, iterations=1)
    return cv2.dilate(binary_img, kernel, iterations=3)

##############################################################################


def extract_train_patches_from_image(img, idx, directory_path, threshold = TRAIN_THRESHOLD, leg_size = TRAIN_LEG_SIZE):
    """
    Extracts training patches from an image and saves them for machine learning training.

    Parameters:
    img (numpy.ndarray): Input image.
    idx (int): Identifier for the image.
    directory_path (str): Directory path to save patches.
    threshold (float, optional): Threshold for meaningful patches. Defaults to TRAIN_THRESHOLD.
    leg_size (int, optional): Vertical stride for patch rows. Defaults to TRAIN_LEG_SIZE.

    """
    global PATCH_ID
    
    img = img.astype(np.float32) / 255.0
    
    
    for leg_y in range(1, 180, leg_size):
      patches_ids = []  
        
      for i in range(NUM_PATCHES_HEIGHT):
          for j in range(NUM_PATCHES_WIDTH):
              start_y = i * PATCH_SIZE + leg_y
              start_x = j * PATCH_SIZE
              end_y = start_y + PATCH_SIZE
              end_x = start_x + PATCH_SIZE
              
              patch = img[start_y:end_y, start_x:end_x]
              if (not is_image_mostly_black(patch, threshold)) and patch.shape == (PATCH_SIZE, PATCH_SIZE):
                  np.save(f"{directory_path}patch-{PATCH_ID}.npy", patch)
                  labels[f'patch-{PATCH_ID}'] = idx
                  patches_ids.append(f'patch-{PATCH_ID}')
                  PATCH_ID += 1
                  
      val_count = int(0.15 * len(patches_ids))
      partition['validation'].extend(patches_ids[-val_count:])
      patches_ids = patches_ids[:-val_count]
      partition['train'].extend(patches_ids)

 
##############################################################################

def extract_test_patches_from_image(img, idx, directory_path, threshold = TEST_THRESHOLD, leg_size = TEST_LEG_SIZE):
    """
    Extracts training patches from an image and saves them for machine learning testing.

    Parameters:
    img (numpy.ndarray): Input image.
    idx (int): Identifier for the image.
    directory_path (str): Directory path to save patches.
    threshold (float, optional): Threshold for meaningful patches. Defaults to TRAIN_THRESHOLD.
    leg_size (int, optional): Vertical stride for patch rows. Defaults to TRAIN_LEG_SIZE.
    
    return: list of test patches
    """
    
    test_patches_ids = []
    global PATCH_ID
    
    
    img = img.astype(np.float32) / 255.0 
    
    
    for leg_y in range(0, 180, leg_size):
      for leg_x in range(1, 180, leg_size):
        for j in range(NUM_PATCHES_WIDTH):
            start_y = leg_y
            start_x = j * PATCH_SIZE + leg_x
            end_y = start_y + PATCH_SIZE
            end_x = start_x + PATCH_SIZE
            
            patch = img[start_y:end_y, start_x:end_x]
            if (not is_image_mostly_black(patch, threshold)) and patch.shape == (PATCH_SIZE, PATCH_SIZE):
                 np.save(f"{directory_path}patch-{PATCH_ID}.npy", patch)
                 labels[f'patch-{PATCH_ID}'] = idx
                 test_patches_ids.append(f'patch-{PATCH_ID}')
                 PATCH_ID += 1

            
    return test_patches_ids
    

##############################################################################




# Check if directory exists
if os.path.exists(patches_directory_path):
    # Clear all files in the directory
    for filename in os.listdir(patches_directory_path):
        file_path = os.path.join(patches_directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
else:
    os.makedirs(patches_directory_path)

partition = dict()
labels = dict()

partition['train'] = []
partition['validation'] = []
partition['test'] = []


mat_file_names = get_files_in_folder(path_to_mat)

mat_file_names = [os.path.join(path_to_mat, fn) for fn in mat_file_names]


test_areas = []  # This list will store pairs (top_test_area, bottom_test_area)

for file in mat_file_names:
    data = loadmat(file)

    top_test_area = data['top_test_area'].flatten()[0]
    bottom_test_area = data['bottom_test_area'].flatten()[0]

    test_areas.append((top_test_area, bottom_test_area))
    
  ##############################################################################  
# List of all your image file names


img_files = get_files_in_folder(path_to_img)
img_files = [os.path.join(path_to_img, fn) for fn in img_files]


# Calculate the number of patches in each dimension
num_patches_height = CROPPED_HEIGHT // PATCH_SIZE
num_patches_width = CROPPED_WIDTH // PATCH_SIZE

temp_img = Image.open(img_files[0])
temp_img = np.array(temp_img)
upper = int(temp_img.shape[0] * 0.20)
lower = int(temp_img.shape[0] * 0.78)


# In order to generate data for the test set take out of comment lines: #254, #256
# And put into comment lines: #253, #255

# Loop over all your image files
for idx, img_file in enumerate(img_files[:PICTURE_SIZE]):
#for idx, img_file in enumerate(img_files[204:]):
  current_top_test_area, current_bottom_test_area = test_areas[idx] 
  #current_top_test_area, current_bottom_test_area = test_areas[idx + 204]
  img = Image.open(img_file)
  train_img_np = np.array(img)
  test_img_np = np.array(img)
  train_img_np[current_top_test_area:current_bottom_test_area + 1, :] = 255

  test_img_np[(0):(current_top_test_area + 1), :] = 255
  test_img_np[(current_bottom_test_area + 1) :, :] = 255

  
  cropped_img_test = test_img_np[current_top_test_area:current_bottom_test_area, :]
  cropped_img_train = train_img_np[upper:lower, :]

  binary_img_train = preprocessing_image(cropped_img_train)
  binary_img_test = preprocessing_image(cropped_img_test)

  extract_train_patches_from_image(binary_img_train, idx, patches_directory_path)
  partition['test'].append(extract_test_patches_from_image(binary_img_test, idx, patches_directory_path))


with open("partition_data.pkl", "wb") as file:
    pickle.dump(partition, file)
    
with open("labels_data.pkl", "wb") as file:
    pickle.dump(labels, file)

    


#%%
# Cell 2: showing the misclassification of the model

misclassified_file_path = r'C:\Users\holtz\Downloads\misclassified_imgs (6).pkl'

with open(misclassified_file_path, 'rb') as f:
    misclassified_imgs = pickle.load(f)
    

# find the image that we misclassified

for pr in misclassified_imgs:
    current_top_test_area_true, current_bottom_test_area_true = test_areas[pr[0]]
    current_top_test_area_false, current_bottom_test_area_false = test_areas[pr[1]]
    true_img = Image.open(img_files[pr[0]])
    false_img = Image.open(img_files[pr[1]])
    true_img_np = np.array(true_img)
    false_img_np = np.array(false_img)
    
   
    cropped_true_img = true_img_np[current_top_test_area_true:current_bottom_test_area_true, :]
    cropped_false_img = false_img_np[current_top_test_area_false:current_bottom_test_area_false, :]
    
    true_file_name = img_files[pr[0]]
    false_file_name = img_files[pr[1]]
    
    true_match = re.search(r'([^\\]+)\.jpg$', true_file_name)
    true_file_name = true_match.group(1)
    
    false_match = re.search(r'([^\\]+)\.jpg$', false_file_name)
    false_file_name = false_match.group(1)

    
    plt.figure(dpi=600)

    # First image: True Test
    plt.subplot(2, 1, 1)  # 1 row, 2 columns, 1st plot
    plt.imshow(cropped_true_img, cmap='gray')
    plt.title(f"True Test: {true_file_name}")
    plt.axis('off')
    
    # Second image: False Predicted Test
    plt.subplot(2, 1, 2)  # 1 row, 2 columns, 2nd plot
    plt.imshow(cropped_false_img, cmap='gray')
    plt.title(f"False Predicted Test: {false_file_name}")
    plt.axis('off')
    
    plt.subplots_adjust(hspace=-0.7) 
    plt.show()


#%%
    
