import splitfolders # type: ignore
import urllib.request
import shutil
import sys
import re
import zipfile
from pathlib import Path

# --- Configuration ---
url = "https://www.kaggle.com/api/v1/datasets/download/ismailpromus/skin-diseases-image-dataset"
zip_file = Path("skin-diseases-image-dataset.zip")
temp_dir = Path(".temp_dataset")
output_dir = Path("dataset")

# --- 1. Download Function ---
def show_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = downloaded * 100 / total_size
        sys.stdout.write(f"\rDownloading: {percent:.1f}% [{downloaded/1024/1024:.1f} / {total_size/1024/1024:.1f} MB]")
        sys.stdout.flush()

# --- 2. Rename Logic ---
def clean_dir_name(name: str) -> str:
    name = re.sub(r'^\d+\.\s*', '', name)                   # Remove leading "10. "
    name = re.sub(r'\s*\([A-Z]+\)', '', name)               # Remove "(BCC)"
    name = re.sub(r'(\s-\s|\s)\d+(\.\d+)?[kK]?$', '', name) # Remove numbers " - 2103"
    name = re.sub(r'\sand\s', ' ', name)                    # Remove " and "
    name = name.lower().strip().replace(' ', '_')           # Lowercase & snake_case
    return name

# --- Execution ---
try:
    # A. Download
    print(f"Starting download to {zip_file}...")
    urllib.request.urlretrieve(url, zip_file, reporthook=show_progress)
    print("\nDownload complete.")

    # B. Unzip
    print("Extracting...")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    # C. Move folders OUT of IMG_CLASSES and delete IMG_CLASSES
    img_classes_path = temp_dir / "IMG_CLASSES"
    
    if img_classes_path.exists():
        print("Flattening directory structure...")
        for subfolder in img_classes_path.iterdir():
            # Move subfolder up one level (from temp/IMG_CLASSES/x to temp/x)
            shutil.move(str(subfolder), str(temp_dir))
        
        # Remove the now empty IMG_CLASSES folder
        img_classes_path.rmdir()

    # D. Rename directories
    print("Renaming directories...")
    for directory in temp_dir.iterdir():
        if directory.is_dir():
            new_name = clean_dir_name(directory.name)
            if new_name != directory.name:
                directory.rename(directory.parent / new_name)

    # E. Split Folders
    print("Splitting dataset...")
    # Clean output dir if exists to prevent errors
    if output_dir.exists():
        shutil.rmtree(output_dir)
        
    splitfolders.ratio(temp_dir, output=str(output_dir), seed=1337, ratio=(.8, .1, .1), group_prefix=None)

    # F. Cleanup
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)
    zip_file.unlink()

    print("Done! Data is ready in 'dataset/'")

except Exception as e:
    print(f"\nError: {e}")
