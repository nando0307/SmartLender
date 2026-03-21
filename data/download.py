"""Download Lending Club dataset from Kaggle."""
import os
import subprocess
import zipfile


def download_lending_club():
    """
    Download the Lending Club dataset using Kaggle CLI.

    Prerequisites:
    - pip install kaggle
    - Place kaggle.json in ~/.kaggle/ with your API credentials
    """
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw')
    os.makedirs(data_dir, exist_ok=True)

    target_file = os.path.join(data_dir, 'accepted_2007_to_2018Q4.csv')

    if os.path.exists(target_file):
        print(f"Dataset already exists at {target_file}")
        return target_file

    print("Downloading Lending Club dataset from Kaggle...")
    print("Make sure you have Kaggle API credentials configured.")
    print("  pip install kaggle")
    print("  Place kaggle.json in ~/.kaggle/")
    print()

    try:
        subprocess.run(
            [
                'kaggle', 'datasets', 'download',
                '-d', 'wordsforthewise/lending-club',
                '-p', data_dir,
            ],
            check=True,
        )
    except FileNotFoundError:
        print("ERROR: kaggle CLI not found. Install it with: pip install kaggle")
        return None
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Kaggle download failed: {e}")
        return None

    # Unzip
    zip_path = os.path.join(data_dir, 'lending-club.zip')
    if os.path.exists(zip_path):
        print("Extracting zip file...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(data_dir)
        os.remove(zip_path)
        print("Extraction complete.")

    if os.path.exists(target_file):
        print(f"Dataset ready at {target_file}")
        return target_file
    else:
        print("WARNING: Expected CSV file not found after extraction.")
        print(f"Files in {data_dir}:")
        for f in os.listdir(data_dir):
            print(f"  {f}")
        return None


if __name__ == '__main__':
    download_lending_club()
