# Model inference code from https://huggingface.co/spaces/ibm-nasa-geospatial/Prithvi-100M-sen1floods11-demo/blob/main/app.py
# Autodistill code from Roboflow

import os

AUTODISTILL_DIR = os.path.expanduser("~/.cache/autodistill")
ORIGINAL_DIR = os.getcwd()

import subprocess
from dataclasses import dataclass

import numpy as np
import rasterio
import supervision as sv
import torch
from autodistill.detection import DetectionBaseModel
from huggingface_hub import hf_hub_download
from mmcv import Config
from mmcv.parallel import collate, scatter

try:
    from mmseg.apis import init_segmentor
    from mmseg.datasets.pipelines import Compose, LoadImageFromFile
except:
    # install mmsegmentation @ https://github.com/open-mmlab/mmsegmentation.git@186572a3ce64ac9b6b37e66d58c76515000c3280#egg=mmsegmentation
    os.chdir(AUTODISTILL_DIR)
    subprocess.run(
        [
            "git",
            "clone",
            "https://github.com/open-mmlab/mmsegmentation.git",
        ]
    )
    os.chdir(os.path.join(AUTODISTILL_DIR, "mmsegmentation"))
    subprocess.run(["git", "checkout", "186572a3ce64ac9b6b37e66d58c76515000c3280"])
    subprocess.run(["pip", "install", "-e", "."])
    os.chdir(ORIGINAL_DIR)

from skimage import exposure

ORIGINAL_DIR = os.getcwd()

if not os.environ.get("HF_TOKEN"):
    raise ValueError(
        "Please set your Hugging Face token as an environment variable called 'HF_TOKEN'"
    )

# download git clone https://github.com/NASA-IMPACT/hls-foundation-os.git
if not os.path.exists(os.path.join(AUTODISTILL_DIR, "hls-foundation-os")):
    os.makedirs(AUTODISTILL_DIR, exist_ok=True)

    os.chdir(AUTODISTILL_DIR)
    subprocess.run(
        [
            "git",
            "clone",
            "--branch",
            "mmseg-only",
            "https://github.com/NASA-IMPACT/hls-foundation-os.git",
        ]
    )

    # pip install -e .
    os.chdir(os.path.join(AUTODISTILL_DIR, "hls-foundation-os"))
    # checkout 9968269915db8402bf4a6d0549df9df57d489e5a
    subprocess.run(["git", "checkout", "9968269915db8402bf4a6d0549df9df57d489e5a"])
    subprocess.run(["pip", "install", "-e", "."])

    # back to original dir
    os.chdir(ORIGINAL_DIR)


def stretch_rgb(rgb):
    ls_pct = 1
    pLow, pHigh = np.percentile(rgb[~np.isnan(rgb)], (ls_pct, 100 - ls_pct))
    img_rescale = exposure.rescale_intensity(rgb, in_range=(pLow, pHigh))

    return img_rescale


def open_tiff(fname):
    with rasterio.open(fname, "r") as src:
        data = src.read()

    return data


def write_tiff(img_wrt, filename, metadata):
    """
    It writes a raster image to file.
    :param img_wrt: numpy array containing the data (can be 2D for single band or 3D for multiple bands)
    :param filename: file path to the output file
    :param metadata: metadata to use to write the raster to disk
    :return:
    """

    with rasterio.open(filename, "w", **metadata) as dest:

        if len(img_wrt.shape) == 2:

            img_wrt = img_wrt[None]

        for i in range(img_wrt.shape[0]):
            dest.write(img_wrt[i, :, :], i + 1)

    return filename


def get_meta(fname):
    with rasterio.open(fname, "r") as src:
        meta = src.meta

    return meta


def preprocess_example(example_list):
    example_list = [os.path.join(os.path.abspath(""), x) for x in example_list]

    return example_list


def inference_segmentor(model, imgs, custom_test_pipeline=None):
    """Inference image(s) with the segmentor.
    Args:
        model (nn.Module): The loaded segmentor.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.
    Returns:
        (list[Tensor]): The segmentation result.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = (
        [LoadImageFromFile()] + cfg.data.test.pipeline[1:]
        if custom_test_pipeline == None
        else custom_test_pipeline
    )

    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = []
    imgs = imgs if isinstance(imgs, list) else [imgs]
    for img in imgs:
        img_data = {"img_info": {"filename": img}}
        img_data = test_pipeline(img_data)
        data.append(img_data)

    data = collate(data, samples_per_gpu=len(imgs))
    if next(model.parameters()).is_cuda:
        # data = collate(data, samples_per_gpu=len(imgs))
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        # img_metas = scatter(data['img_metas'],'cpu')
        # data['img_metas'] = [i.data[0] for i in data['img_metas']]

        img_metas = data["img_metas"].data[0]
        img = data["img"]
        data = {"img": img, "img_metas": img_metas}

    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


def process_test_pipeline(custom_test_pipeline, bands=None):
    # change extracted bands if necessary
    if bands is not None:

        extract_index = [
            i for i, x in enumerate(custom_test_pipeline) if x["type"] == "BandsExtract"
        ]

        if len(extract_index) > 0:

            custom_test_pipeline[extract_index[0]]["bands"] = eval(bands)

    collect_index = [
        i for i, x in enumerate(custom_test_pipeline) if x["type"].find("Collect") > -1
    ]

    # adapt collected keys if necessary
    if len(collect_index) > 0:
        keys = [
            "img_info",
            "filename",
            "ori_filename",
            "img",
            "img_shape",
            "ori_shape",
            "pad_shape",
            "scale_factor",
            "img_norm_cfg",
        ]
        custom_test_pipeline[collect_index[0]]["meta_keys"] = keys

    return custom_test_pipeline


HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class HLSGeospatial(DetectionBaseModel):
    def __init__(self, model_name: str):
        # adding this until we build support for the other models
        if model_name != "Prithvi-100M-sen1floods11":
            raise ValueError(f"Model name {model_name} not supported")

        if model_name == "Prithvi-100M-sen1floods11":
            config_path = hf_hub_download(
                repo_id="ibm-nasa-geospatial/Prithvi-100M-sen1floods11",
                filename="sen1floods11_Prithvi_100M.py",
                token=os.environ.get("HF_TOKEN"),
            )
            ckpt = hf_hub_download(
                repo_id="ibm-nasa-geospatial/Prithvi-100M-sen1floods11",
                filename="sen1floods11_Prithvi_100M.pth",
                token=os.environ.get("HF_TOKEN"),
            )

        elif model_name == "Prithvi-100M-burn-scar":
            config_path = hf_hub_download(
                repo_id="ibm-nasa-geospatial/Prithvi-100M-burn-scar",
                filename="burn_scars_Prithvi_100M.py",
                token=os.environ.get("token"),
            )
            ckpt = hf_hub_download(
                repo_id="ibm-nasa-geospatial/Prithvi-100M-burn-scar",
                filename="burn_scars_Prithvi_100M.pth",
                token=os.environ.get("token"),
            )
        elif model_name == "Prithvi-100M-multi-temporal-crop-classification":
            config_path = hf_hub_download(
                repo_id="ibm-nasa-geospatial/Prithvi-100M-multi-temporal-crop-classification",
                filename="multi_temporal_crop_classification_Prithvi_100M.py",
                token=os.environ.get("token"),
            )
            ckpt = hf_hub_download(
                repo_id="ibm-nasa-geospatial/Prithvi-100M-multi-temporal-crop-classification",
                filename="multi_temporal_crop_classification_Prithvi_100M.pth",
                token=os.environ.get("token"),
            )
        else:
            raise ValueError(f"Model name {model_name} not supported")

        os.chdir(os.path.join(AUTODISTILL_DIR, "hls-foundation-os"))

        config = Config.fromfile(config_path)
        config.model.backbone.pretrained = None

        # change to dir of this file
        os.chdir(ORIGINAL_DIR)

        self.config = config

        model = init_segmentor(config, ckpt, device=DEVICE)
        custom_test_pipeline = process_test_pipeline(model.cfg.data.test.pipeline, None)

        self.model = model
        self.custom_test_pipeline = custom_test_pipeline

    def predict(self, input: str) -> sv.Detections:
        # open image
        input = preprocess_example([input])[0]

        result = inference_segmentor(self.model, input, self.custom_test_pipeline)

        mask = open_tiff(input)
        rgb = stretch_rgb(
            (mask[[3, 2, 1], :, :].transpose((1, 2, 0)) / 10000 * 255).astype(np.uint8)
        )
        meta = get_meta(input)
        mask = np.where(mask == meta["nodata"], 1, 0)
        mask = np.max(mask, axis=0)[None]
        rgb = np.where(mask.transpose((1, 2, 0)) == 1, 0, rgb)
        rgb = np.where(rgb < 0, 0, rgb)
        rgb = np.where(rgb > 255, 255, rgb)

        # convert to boolean mask
        prediction = np.where(mask == 1, 0, result[0] * 255)

        xyxy = sv.detection.utils.mask_to_xyxy(prediction)

        # convert explicitly to True and False values
        prediction = prediction.astype(bool)

        return sv.Detections(
            xyxy=xyxy,
            confidence=np.array([1.0]),
            class_id=np.array([0]),
            mask=prediction,
        )
