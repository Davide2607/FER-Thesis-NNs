import os
import subprocess
import zipfile

def download_and_extract():
    # Define the Google Drive folder ID and output paths
    folder_id = "1tkWnzQVDiWGN3Ofjf33IbL-ZxSr-2ncE"
    zip_file = "data.zip"
    output_folder = "data"

    # Ensure gdown is installed
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        subprocess.check_call(["pip", "install", "gdown"])

    # Download the folder as a zip file
    print("Downloading data folder...")
    gdown_url = f"https://drive.google.com/uc?id={folder_id}&export=download"
    subprocess.check_call(["gdown", gdown_url, "-O", zip_file])

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Extract the zip file
    print("Extracting data...")
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(output_folder)

    # Clean up the zip file
    os.remove(zip_file)
    print(f"Data downloaded and extracted to '{output_folder}'.")

if __name__ == "__main__":
    download_and_extract()