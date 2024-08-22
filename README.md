# BOP Toolkit Extended with SAM2 for Object Visibility Estimation on HOT3D

<p align="center">
  <img src="docs/result_aria_clip_1849_obj_2.gif" height="320" />
  <img src="docs/result_quest3_clip_100_obj_1.gif" height="320" /> 
</p>


This repository extends the original [BOP Toolkit](https://github.com/thodan/bop_toolkit) with code for object visibility estimation based on [Segment Anything 2 (SAM2)](https://github.com/facebookresearch/segment-anything-2), specifically for the [HOT3D dataset](https://github.com/facebookresearch/hot3d).
### Contributors
- [Taeyeop Lee](https://sites.google.com/view/taeyeop-lee)
- [Tomas Hodan](https://cmp.felk.cvut.cz/~hodanto2/)
- [Prithviraj Banerjee](https://www.linkedin.com/in/prithvirajb/)
- [Van Nguyen Nguyen](https://nv-nguyen.github.io/)

### 1. Environment Setup
We recommend setting up an environment compatible with SAM2, which requires recent CUDA and PyTorch (>=2.3.1) versions.

    conda create -n bop_sam2 python=3.10.14
    conda activate bop_sam2
    pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

### 2. Download SAM2 Weights
    cd segment_anything_v2
    pip install -e .
    cd checkpoints
    ./download_ckpts.sh
    

### 3. Install Dependencies 
    sudo apt-get install imagemagick
    pip install -r requirements.txt
    


### 4. Download HOT3D Dataset. 
We tested the `train_aria`, `train_quest3`, and `object_models_eval` from [Hugging Face (bop-benchmark/hot3d)](https://huggingface.co/datasets/bop-benchmark/datasets/tree/main/hot3d)


### 5. Preprocess Data 
Save RGB image, render mask, and SAM2 estimation results for video estimation. We use the `sam2_hiera_large.pt` model. 

    cd sam2_hot3d
    python preprocess.py --clips_dir /path/to/hot3d/train_quest3 --object_models_dir /path/to/hot3d/object_models --output_dir ../output/ --clip_start 0 --clip_end -1
    

### 6. Generate Visibility Masks
We use the `sam2_hiera_tiny.pt` model for efficient SAM2 video inference. To visualize the results, use the `--debug` flag.

    cd sam2_hot3d
    python run_video.py --clips_dir /path/to/hot3d/train_quest3 --object_models_dir /path/to/hot3d/object_models --output_dir ../output/ --clip_start 0 --clip_end -1 --debug --conf_thres 80 --iou_thres 60
    
#### 7. (Optional) Customizing Parameters
You can select different models and adjust confidence levels in `run_video.py` according to your requirements: 
```
--conf_thres = 80  # Range: 0 ~ 100
--iou_thres = 60  # Range: 0 ~ 100
```
For efficiency, we use `sam2_hiera_large.pt` in `preprocess.py` and `sam2_hiera_tiny.pt` in `run_video.py`. Feel free to experiment with different models based on your needs.

### Acknowledgement
The code is adapted from [BOP](https://github.com/thodan/bop_toolkit), [SAM2](https://github.com/facebookresearch/segment-anything-2), [HOT3D](https://github.com/facebookresearch/hot3d).
### Contact
If you have any question, feel free to create an issue or contact the [Taeyeop Lee](https://sites.google.com/view/taeyeop-lee/).

