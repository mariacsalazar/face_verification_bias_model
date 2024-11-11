import os
import random
import shutil
from math import floor

def copy_with_overwrite(src, dst):
    shutil.copy(src, dst)

def check_empty_folders(parent_folder):
    for img_num_folder in os.listdir(parent_folder):
        img_num_folder_path = os.path.join(parent_folder, img_num_folder)

        # Check if it is a directory and contains no files
        if os.path.isdir(img_num_folder_path):
            # List all files (not subdirectories) in the current img_num_folder
            files = [f for f in os.listdir(img_num_folder_path) if os.path.isfile(os.path.join(img_num_folder_path, f))]
                
            if not files:  # No files found in img_num_folder
                print(f"Empty folder: {img_num_folder_path}")

def split_folders(original_folder, target_folder, smaller_foldersize, train_percentage, test_percentage, validation_percentage):
    """
    Splits a set of image folders into train, test, and validation sets based on the specified percentages.
    
    Args:
        original_folder (str): Path to the original folder containing all image folders.
        target_folder (str): Path to the target folder where train, test, and validation folders will be created.
        smaller_foldersize (int): The number of folders to be selected for splitting.
        train_percentage (float): Percentage of images to allocate to the train set.
        test_percentage (float): Percentage of images to allocate to the test set.
        validation_percentage (float): Percentage of images to allocate to the validation set.
    
    Raises:
        ValueError: If the percentages for train, test, and validation do not sum to 1.
    """

    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)

    os.makedirs(target_folder, exist_ok=True)

    epsilon = 1e-9
    if abs(train_percentage + test_percentage + validation_percentage - 1) > epsilon:
        raise ValueError("Train, test, and validation percentage must sum to 1")

    # Create directories for train, test, validation
    train_folder = os.path.join(target_folder, "train")
    test_folder = os.path.join(target_folder, "test")
    val_folder = os.path.join(target_folder, "validation") if validation_percentage > 0 else None

    # Create the directories again since the target folder was cleared
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    if val_folder:
        os.makedirs(val_folder, exist_ok=True)

    # Get all class folders (0 to 10571 in our case)
    all_folders = [f for f in os.listdir(original_folder) if os.path.isdir(os.path.join(original_folder, f))]
    all_folders.sort()

    # Randomly select the required number of folders
    selected_folders = random.sample(all_folders, smaller_foldersize)

    # Copy images into train, test, and validation for all selected folders
    for folder in selected_folders:
        folder_path = os.path.join(original_folder, folder)
        images = os.listdir(folder_path)
        random.shuffle(images)
        
        # Calculate the number of images for each set based on percentages
        num_images = len(images)

        if num_images >= 3:
            # Ensure each set has at least one image
            train_count = max(1, floor(num_images * train_percentage))
            test_count = max(1, floor(num_images * test_percentage))
            validation_count = max(1, num_images - train_count - test_count)

            # Adjust if the sum exceeds the total number of images
            if train_count + test_count + validation_count > num_images:
                # Reduce train_count if necessary to balance
                train_count = num_images - test_count - validation_count
        else:
            # If fewer than 3 images, prioritize train set, then test, then validation
            train_count = 1
            test_count = 1 if num_images > 1 else 0
            validation_count = 1 if num_images > 2 else 0

        # Assign images to train, test, and validation
        train_images = images[:train_count]
        test_images = images[train_count:train_count + test_count]
        if val_folder:
            val_images = images[train_count + test_count:train_count + test_count + validation_count]

        # Create folder structure in train, test, validation
        os.makedirs(os.path.join(train_folder, folder), exist_ok=True)
        os.makedirs(os.path.join(test_folder, folder), exist_ok=True)
        if val_folder and validation_count > 0:
            os.makedirs(os.path.join(val_folder, folder), exist_ok=True)

        # Copy the images into the corresponding folders
        for img in train_images:
            src = os.path.join(folder_path, img)
            dst = os.path.join(train_folder, folder, img)
            copy_with_overwrite(src, dst)

        for img in test_images:
            src = os.path.join(folder_path, img)
            dst = os.path.join(test_folder, folder, img)
            copy_with_overwrite(src, dst)

        if val_folder and validation_count > 0:
            for img in val_images:
                src = os.path.join(folder_path, img)
                dst = os.path.join(val_folder, folder, img)
                copy_with_overwrite(src, dst)

    print("Completed folder creation and test-train-validation split with {} classes.".format(smaller_foldersize))
        
    check_empty_folders(train_folder)
    check_empty_folders(test_folder)
    if val_folder:
        check_empty_folders(val_folder)


def main():
    # Paths to 'imgs' and 'imgs_subset' folders in the parent directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
    original_folder = os.path.join(parent_dir, 'data/imgs')
    target_folder = os.path.join(parent_dir, 'data/imgs_subset')

    # Set smaller_foldersize and percentages for splitting
    total_classes = 500 # Ensure it is between 0 to 10572 for our dataset
    train_percentage = 0.8   
    test_percentage = 0.1  
    validation_percentage = 0.1  # In case you do not need a validation set, set it to 0

    split_folders(original_folder, target_folder, total_classes, train_percentage, test_percentage, validation_percentage)

if __name__ == "__main__":
    main()