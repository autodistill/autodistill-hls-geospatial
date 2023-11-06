<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png"
      >
    </a>
  </p>
</div>

# Autodistill HLS Geospatial Module

This repository contains the code supporting the HLS Geospatial base model for use with [Autodistill](https://github.com/autodistill/autodistill).

[Harmonized Landsat and Sentinel-2 (HLS) Prithvi](https://github.com/NASA-IMPACT/hls-foundation-os) is a collection of foundation models for geospatial analysis, developed by NASA and IBM. You can use Autodistill to automatically label images for use in training segmentation models.

The following models are supported:

- [Prithvi-100M-sen1floods11](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M-sen1floods11)

This module accepts `tiff` files as input and returns segmentation masks.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [HLS Geospatial Autodistill documentation](https://autodistill.github.io/autodistill/base_models/clip/).

## Installation

To use HLS Geospatial with autodistill, you need to install the following dependency:

```bash
pip3 install autodistill-hls-geospatial
```

## Quickstart

```python
from autodistill_hls_geospatial import HLSGeospatial

# define an ontology to map class names to our HLS Geospatial prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
import numpy as np
import rasterio
from skimage import exposure
import supervision as sv

from autodistill_hls_geospatial import HLSGeospatial

def stretch_rgb(rgb):
    ls_pct = 1
    pLow, pHigh = np.percentile(rgb[~np.isnan(rgb)], (ls_pct, 100 - ls_pct))
    img_rescale = exposure.rescale_intensity(rgb, in_range=(pLow, pHigh))

    return img_rescale


# replace with the file you want to use
with rasterio.open("USA_430764_S2Hand.tif") as src:
    image = src.read()

    mask = image

    rgb = stretch_rgb(
        (mask[[3, 2, 1], :, :].transpose((1, 2, 0)) / 10000 * 255).astype(np.uint8)
    )

    base_model = HLSGeospatial()

    # replace with the file you want to use
    detections = base_model.predict("USA_430764_S2Hand.tif")

    mask_annotator = sv.MaskAnnotator()

    annotated_image = mask_annotator.annotate(scene=rgb, detections=detections)

    sv.plot_image(annotated_image, size=(10, 10))

# label a folder of .tif files
base_model.label("./context_images", extension=".tif")
```


## License

This project is licensed under an [Apache 2.0 license](LICENSE).

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!