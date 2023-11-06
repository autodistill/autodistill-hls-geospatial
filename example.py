
from autodistill_hls_geospatial import HLSGeospatial
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


#replace with the name of the file you want to label
FILE_NAME = "USA_430764_S2Hand.tif"

with rasterio.open(FILE_NAME) as src:
    image = src.read()

    mask = image

    rgb = stretch_rgb(
        (mask[[3, 2, 1], :, :].transpose((1, 2, 0)) / 10000 * 255).astype(np.uint8)
    )

    base_model = HLSGeospatial()

    # replace with the file you want to use
    detections = base_model.predict(FILE_NAME)

    mask_annotator = sv.MaskAnnotator()

    annotated_image = mask_annotator.annotate(scene=rgb, detections=detections)

    sv.plot_image(annotated_image, size=(10, 10))

# label a folder of .tif files
base_model.label("./context_images", extension=".tif")