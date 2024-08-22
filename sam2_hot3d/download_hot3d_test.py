import os
import subprocess


def download_clips(start_id, end_id, dataset_name):
    base_url = "https://huggingface.co/datasets/bop-benchmark/datasets/resolve/main/hot3d"
    base_dir = "./dataset"

    # Create dataset directory if it doesn't exist
    dataset_dir = os.path.join(base_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    for clip_id in range(start_id, end_id + 1):
        # for idx in range(1,3,1):
            # clip_filename = f"obj_{clip_id:06d}_{idx}.mp4"
        clip_filename = f"obj_{clip_id:06d}.glb"
        url = f"{base_url}/{dataset_name}/{clip_filename}"

        # Use wget to download the file
        command = f"wget -P {dataset_dir} {url}"

        try:
            subprocess.run(command, shell=True, check=True)
            print(f"Downloaded: {clip_filename} to {dataset_dir}")
        except subprocess.CalledProcessError:
            print(f"Failed to download: {clip_filename}")

    clip_filename = f"models_info.json"
    url = f"{base_url}/{dataset_name}/{clip_filename}"

    # Use wget to download the file
    command = f"wget -P {dataset_dir} {url}"

    try:
        subprocess.run(command, shell=True, check=True)
        print(f"Downloaded: {clip_filename} to {dataset_dir}")
    except subprocess.CalledProcessError:
        print(f"Failed to download: {clip_filename}")


# Download test_quest3 dataset
# print("Downloading test_quest3 dataset...")
# download_clips(1288, 1848, "test_quest3")

# Download test_aria dataset
# print("Downloading test_aria dataset...")
# download_clips(3365, 3831, "test_aria")
# download_clips(1, 33, "object_ref_quest3_dynamic")
# download_clips(1, 33, "object_ref_quest3_static")
# download_clips(1, 33, "object_ref_quest3_static_vis")
# download_clips(1, 33, "object_ref_quest3_dynamic_vis")
download_clips(1, 33, "object_models_eval")

print("Download completed.")