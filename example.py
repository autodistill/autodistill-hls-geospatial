import numpy as np
import rasterio
from skimage import exposure

from autodistill_hls_geospatial import HLSGeospatial

# define an ontology to map class names to our HLS Geospatial prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model



def stretch_rgb(rgb):
    ls_pct = 1
    pLow, pHigh = np.percentile(rgb[~np.isnan(rgb)], (ls_pct, 100 - ls_pct))
    img_rescale = exposure.rescale_intensity(rgb, in_range=(pLow, pHigh))

    return img_rescale


with rasterio.open("USA_430764_S2Hand.tif") as src:
    image = src.read()

    mask = image

    rgb = stretch_rgb(
        (mask[[3, 2, 1], :, :].transpose((1, 2, 0)) / 10000 * 255).astype(np.uint8)
    )

    base_model = HLSGeospatial()

    detections = base_model.predict("USA_430764_S2Hand.tif")

    import supervision as sv

    mask_annotator = sv.MaskAnnotator()

    annotated_image = mask_annotator.annotate(scene=rgb, detections=detections)

    sv.plot_image(annotated_image, size=(10, 10))