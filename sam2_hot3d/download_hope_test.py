import os
import subprocess
import zipfile


def download_and_unzip_hope_dataset(save_dir):
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Dataset list and URL
    base_url = "https://huggingface.co/datasets/bop-benchmark/datasets/resolve/main/hope"
    datasets = [
        "hope_base.zip",
        "hope_models.zip",
        "hope_onboarding_dynamic.zip",
        "hope_onboarding_static.zip",
        "hope_test_bop24.zip",
        "hope_test_realsense.zip",
        "hope_test_vicon.zip",
        # "hope_train_pbr.z01",
        # "hope_train_pbr.z02",
        # "hope_train_pbr.zip",
        "hope_val_realsense.zip"
    ]

    # Download and unzip datasets
    for dataset in datasets:
        url = f"{base_url}/{dataset}"
        output_path = os.path.join(save_dir, dataset)

        print(f"Downloading {dataset}...")
        command = f"wget -O {output_path} {url}"

        try:
            subprocess.run(command, shell=True, check=True)
            print(f"Downloaded {dataset} to {save_dir}")

            # Unzip
            print(f"Unzipping {dataset}...")
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(save_dir)
            print(f"Unzipped {dataset}")

            # Remove downloaded zip file (optional)
            os.remove(output_path)
            print(f"Removed {dataset}")

        except subprocess.CalledProcessError:
            print(f"Failed to download: {dataset}")
        except zipfile.BadZipFile:
            print(f"Failed to unzip: {dataset}. File may be corrupted.")

    print("All datasets download and unzip attempts completed.")


# Usage example:
save_directory = "./dataset"
download_and_unzip_hope_dataset(save_directory)