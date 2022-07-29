# Detectron2

    Mask RCNN, Pointrend, Faster RCNN and evaluation
    
    
# Dependencies:
    pip3 install -r requirements.txt
    
    pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/{CUDA_VERSION}/{TORCH_VERSION}/index.html
    
    git clone --branch v0.6 https://github.com/facebookresearch/detectron2.git detectron2_repo
    
    pip install -e detectron2_repo  (If have to compile from source) (optional)
    
    Version used locally (Cuda 11.7) - 
    python3 -m pip install detectron2==0.6 -f   https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.10/index.html
    
    
Steps:
    Download Kitti_official_semantic from the list of datasets in the shared folder on the system 10.165.81.147
    change path names in code wherever necessary
    
    run dependencies based on your system
    
    python3 Mask_rcnn.py             To Run mask rcnn model through detectron2
    python3 faster_rcnn.py           To Run faster rcnn model through detectron2
    python3 masks.py                 To run and get masks for predictions and ground truth  (preds and gt)
    python3 masks2.py                To run and get masks for predictions and ground truth for pedestrians and vehicles (grey scale images)
    python3 pointrend.py             To Run pointrend image segmentation model through detectron2
    python3 ptrend_masks.py          To generate pointrend masks and download masks numpy in .npy files
    python3 ensemble.py              To get the ensemble of mask rcnn and pointrend masks stored as .npy files
    
