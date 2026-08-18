"""Shared constants for training (Predict2.5 & Transfer2.5)."""

from typing import Dict, Final, Tuple

# ============================================================
# Dataset Paths
# ============================================================

NSCLC_DATASET_PATH = "data/NSCLC/processed/"
MOSMED_DATASET_PATH = "data/MOSMED/processed/"
MELA2022_DATASET_PATH = "data/MELA2022/raw/"
VINDR_DATASET_PATH = "data/VinDr/v1/processed/"
TCIA_DATASET_PATH = "data/TCIA/"

# ============================================================
# Cosmos-Reason1 Text Encoder
# ============================================================

CR1_HIDDEN_SIZE = 3584
CR1_NUM_LAYERS = 28
CR1_MAX_LENGTH = 512
CR1_FULL_CONCAT_DIM = CR1_HIDDEN_SIZE * CR1_NUM_LAYERS
CR1_EMBEDDING_DIM = CR1_FULL_CONCAT_DIM

CROSSATTN_EMB_CHANNELS = 1024
CROSSATTN_PROJ_IN_CHANNELS = CR1_FULL_CONCAT_DIM
CROSSATTN_PROJ_OUT_CHANNELS = 1024

# ============================================================
# Volume / Image Configuration
# ============================================================

VOL_SIZE = 256
IMG_HEIGHT = VOL_SIZE
IMG_WIDTH = VOL_SIZE

# ============================================================
# Frame Configuration
# ============================================================

NUM_FRAMES = 93
NUM_LATENT_FRAMES = 24  # 1 + (93-1)//4 = 24

FRAME_HEIGHT = IMG_HEIGHT
FRAME_WIDTH = IMG_WIDTH

# ============================================================
# Checkpoint UUIDs
# ============================================================

COSMOS_TOKENIZER_UUID = "685afcaa-4de2-42fe-b7b9-69f7a2dee4d8"
COSMOS_2B_PRETRAINED_UUID = "d20b7120-df3e-4911-919d-db6e08bad31c"
COSMOS_REASON_UUID = "cb3e3ffa-7b08-4c34-822d-61c7aa31a14f"
T5_11B_UUID = "4dbf13c6-1d30-4b02-99d6-75780dd8b744"

# ============================================================
# 7-View Multiview Camera Configuration
# ============================================================

XRAY_CAMERAS: Final[Tuple[str, ...]] = (
    "xray_ap",
    "xray_lateral_right",
    "xray_rao",
    "xray_pa",
    "xray_lao",
    "xray_lateral_left",
    "xray_cranial",
)

XRAY_VIEW_MAPPING: Final[Dict[str, int]] = dict(
    zip(XRAY_CAMERAS, range(len(XRAY_CAMERAS)))
)

XRAY_EXTRINSICS: Final[Dict[str, Dict[str, float]]] = {
    "xray_ap":            {"azimuth":   0.0, "elevation":  0.0},
    "xray_lateral_right": {"azimuth": 270.0, "elevation":  0.0},
    "xray_rao":           {"azimuth": 315.0, "elevation":  0.0},
    "xray_pa":            {"azimuth": 180.0, "elevation":  0.0},
    "xray_lao":           {"azimuth":  45.0, "elevation":  0.0},
    "xray_lateral_left":  {"azimuth":  90.0, "elevation":  0.0},
    "xray_cranial":       {"azimuth":   0.0, "elevation": 30.0},
}

XRAY_FOV_RANGE: Final[Tuple[float, float]] = (10.0, 14.0)
XRAY_FOV_DEFAULT: Final[float] = (XRAY_FOV_RANGE[0] + XRAY_FOV_RANGE[1]) / 2.0
XRAY_DISTANCE_RANGE: Final[Tuple[float, float]] = (7.875, 8.125)
XRAY_DISTANCE_DEFAULT: Final[float] = (XRAY_DISTANCE_RANGE[0] + XRAY_DISTANCE_RANGE[1]) / 2.0

XRAY_CAPTION_PREFIXES: Final[Dict[str, str]] = {
    "xray_ap": (
        "A chest radiograph in anteroposterior (AP) projection."
        " The X-ray beam travels from anterior to posterior through the thorax."
        " The cardiac silhouette appears enlarged compared to PA: the heart sits anteriorly,"
        " far from the posterior detector, producing greater geometric magnification."
        " Bilateral lung fields fan outward symmetrically with horizontal rib arches."
    ),
    "xray_lateral_right": (
        "A chest radiograph in right lateral projection."
        " The X-ray beam traverses the thorax from the patient's right side to the left."
        " The thoracic spine projects as a bright posterior column;"
        " ribs from both sides overlap, and the cardiac shadow is oval-shaped"
        " with radiolucent retrosternal and retrocardiac spaces."
    ),
    "xray_rao": (
        "A chest radiograph in right anterior oblique (RAO) projection at 45 degrees."
        " The X-ray beam passes obliquely from the right-anterior direction through the chest."
        " The cardiac silhouette shifts rightward; the right cardiac border and right atrium are prominent."
        " The spine projects posteriorly toward the right half of the field."
    ),
    "xray_pa": (
        "A chest radiograph in posteroanterior (PA) projection."
        " The X-ray beam travels from posterior to anterior, minimizing cardiac magnification."
        " The heart appears compact with sharp mediastinal and cardiac borders."
        " Both lung fields are fully expanded and symmetrically visible."
    ),
    "xray_lao": (
        "A chest radiograph in left anterior oblique (LAO) projection at 45 degrees."
        " The X-ray beam passes obliquely from the left-anterior direction through the chest."
        " The aortic arch and aortic knob are opened and prominently displayed."
        " The cardiac silhouette shifts leftward with the left atrium visible on the posterior cardiac border."
    ),
    "xray_lateral_left": (
        "A chest radiograph in left lateral projection."
        " The X-ray beam traverses the thorax from the patient's left side to the right."
        " The thoracic spine projects as a bright posterior column;"
        " the left hemidiaphragm merges anteriorly with the inferior cardiac border."
    ),
    "xray_cranial": (
        "A chest radiograph in cranially-angulated AP projection with a 30-degree superior tilt."
        " The X-ray beam enters the thorax from the superior-anterior direction and exits inferiorly."
        " The aortic arch and superior mediastinum are clearly displayed."
        " Clavicles project superiorly over the lung apices in this angulated view."
    ),
}

XRAY_PROMPT_TEMPLATE: Final[str] = (
    "{prefix}"
    " This is a grayscale digitally reconstructed radiograph (DRR) computed by"
    " object-centric diverging-ray forward projection of a 3D CT volume."
    " The thoracic CT volume is fixed at the world origin (isocenter);"
    " the virtual point X-ray source orbits around it at source-to-isocenter distance {distance:.2f}"
    " (scene units), cone-beam full field of view {fov:.1f} degrees,"
    " azimuth {azimuth:.1f} deg, elevation {elevation:.1f} deg."
    " CT Hounsfield-unit voxel values serve as linear attenuation proxies:"
    " they are accumulated via a weighted ray-projection integral along each diverging ray"
    " from source depth {znear:.2f} to {zfar:.2f} (scene units),"
    " a window that fully brackets the isocenter at depth {distance:.2f}."
    " The projection is normalised by its maximum so the densest structure maps to white:"
    " cortical bone and calcifications are bright white;"
    " soft tissue — myocardium, great vessels, diaphragm — renders as intermediate grays;"
    " air-filled alveolar lung parenchyma has near-zero accumulated density and appears dark."
    " The output is strictly monochrome grayscale — not a color photograph, not an MRI,"
    " and free of motion blur or film grain."
)

NUM_XRAY_VIEWS: Final[int] = len(XRAY_CAMERAS)

CONTROL_SIGNAL_TYPES: Final[Tuple[str, ...]] = (
    "seg_mask",
    "edge_map",
    "depth_map",
)
