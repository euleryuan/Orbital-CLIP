import os
import random
import shutil
import logging
from typing import List, Optional

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def split_orbital_dataset(
    train_dir: str, 
    val_dir: str, 
    split_ratio: float = 0.2, 
    random_seed: int = 42,
    file_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.dcm')
) -> None:
    """
    Partitions the orbital imaging dataset into training and validation subsets.
    
    This function implements a stratified split by class to ensure that rare 
    orbital diseases are represented in both subsets proportional to their 
    original distribution.

    Args:
        train_dir (str): Path to the source training directory organized by class.
        val_dir (str): Destination path for the validation subset.
        split_ratio (float): Proportion of the dataset to be moved to the validation set.
        random_seed (int): Seed for reproducibility of the pseudo-random partition.
        file_extensions (tuple): Supported image formats (including DICOM for medical use).
    """
    
    # Ensuring environmental reproducibility
    random.seed(random_seed)
    
    if not os.path.exists(train_dir):
        logger.error(f"Source directory {train_dir} does not exist.")
        return

    # Create destination directory if it doesn't exist
    os.makedirs(val_dir, exist_ok=True)

    # Systematic iteration through disease categories
    categories = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    logger.info(f"Identified {len(categories)} disease categories for partitioning.")

    for category in categories:
        class_train_path = os.path.join(train_dir, category)
        class_val_path = os.path.join(val_dir, category)

        # Extraction of image assets
        all_images = [
            f for f in os.listdir(class_train_path) 
            if f.lower().endswith(file_extensions)
        ]
        
        num_images = len(all_images)
        if num_images == 0:
            logger.warning(f"Category '{category}' is empty. Skipping...")
            continue

        os.makedirs(class_val_path, exist_ok=True)

        # Randomized selection for the validation cohort
        val_count = int(num_images * split_ratio)
        val_subset = random.sample(all_images, val_count)

        logger.info(f"Partitioning category '{category}': Total={num_images}, Val_Size={val_count}")

        # Executing the migration with metadata synchronization
        for img_filename in val_subset:
            # 1. Image migration
            src_img = os.path.join(class_train_path, img_filename)
            dst_img = os.path.join(class_val_path, img_filename)
            shutil.move(src_img, dst_img)

            # 2. Associated Metadata/Label migration (e.g., Clinical Reports or Labels)
            # We assume label files share the same basename as images
            base_name = os.path.splitext(img_filename)[0]
            for label_ext in ['.txt', '.json', '.xml']:
                label_filename = base_name + label_ext
                src_label = os.path.join(class_train_path, label_filename)
                if os.path.exists(src_label):
                    dst_label = os.path.join(class_val_path, label_filename)
                    shutil.move(src_label, dst_label)

    logger.info("Dataset stratification and partitioning completed successfully.")

if __name__ == "__main__":
    # Standardizing paths for the experiment
    DATASET_ROOT = "dataset_orbital_diseases"
    TRAIN_PATH = os.path.join(DATASET_ROOT, "train")
    VAL_PATH = os.path.join(DATASET_ROOT, "val")

    # Execute the split
    split_orbital_dataset(
        train_dir=TRAIN_PATH,
        val_dir=VAL_PATH,
        split_ratio=0.2,
        random_seed=2024 # Custom seed for the final study run
    )
