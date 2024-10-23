import os
import random
import shutil
from math import floor

def copy_with_overwrite(src, dst):
    shutil.copy(src, dst)

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
    val_folder = os.path.join(target_folder, "validation")

    # Create the directories again since the target folder was cleared
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
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

        # Shuffle the images randomly
        random.shuffle(images)

        # Calculate number of images for train, test, validation based on percentages
        train_count = floor(len(images) * train_percentage)
        test_count = floor(len(images) * test_percentage)

        train_images = images[:train_count]
        test_images = images[train_count:train_count + test_count]
        val_images = images[train_count + test_count:]

        # Create folder structure in train, test, validation
        os.makedirs(os.path.join(train_folder, folder), exist_ok=True)
        os.makedirs(os.path.join(test_folder, folder), exist_ok=True)
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

        for img in val_images:
            src = os.path.join(folder_path, img)
            dst = os.path.join(val_folder, folder, img)
            copy_with_overwrite(src, dst)

    print("Completed folder creation and test-train-validation split with {} classes.".format(smaller_foldersize))

if __name__ == "__main__":
    # Paths to 'imgs' and 'imgs_subset' folders in the parent directory
    original_folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'imgs'))
    target_folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'imgs_subset'))

    # Set smaller_foldersize and percentages for splitting
    total_classes = 5 # Ensure it is between 0 to 10572 for our dataset
    train_percentage = 0.8   
    test_percentage = 0.1    
    validation_percentage = 0.1  # In case you do not need a validation set, set it to 0

    split_folders(original_folder, target_folder, total_classes, train_percentage, test_percentage, validation_percentage)
