"""Constants for predict2_5 package."""

from shared.constants import (
    CR1_EMBEDDING_DIM,
    CR1_FULL_CONCAT_DIM,
    CR1_HIDDEN_SIZE,
    CR1_MAX_LENGTH,
    CR1_NUM_LAYERS,
    CROSSATTN_EMB_CHANNELS,
    CROSSATTN_PROJ_IN_CHANNELS,
    CROSSATTN_PROJ_OUT_CHANNELS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    IMG_HEIGHT,
    IMG_WIDTH,
    NUM_FRAMES,
    NUM_LATENT_FRAMES,
    NUM_XRAY_VIEWS,
    VOL_SIZE,
    COSMOS_2B_PRETRAINED_UUID,
    COSMOS_REASON_UUID,
    COSMOS_TOKENIZER_UUID,
    T5_11B_UUID,
    XRAY_CAMERAS,
    XRAY_CAPTION_PREFIXES,
    XRAY_DISTANCE_DEFAULT,
    XRAY_DISTANCE_RANGE,
    XRAY_EXTRINSICS,
    XRAY_FOV_DEFAULT,
    XRAY_FOV_RANGE,
    XRAY_PROMPT_TEMPLATE,
    XRAY_VIEW_MAPPING,
)

DEFAULT_CT_DIRS = [
    "data/NSCLC/processed/",
    "data/MELA2022/raw/",
    "data/TCIA/",
]

DEFAULT_XR_DIR = "data/VinDr/v1/processed/"
DEFAULT_EMBEDDING_DIR = "cosmos_reason_embeddings"
CR1_HF_REVISION = "3210bec0495fdc7a8d3dbb8d58da5711eab4b423"

PROMPTS = [(
    "A 360-degree rotational view of a chest CT scan showing anatomical "
    "structures from all angles, rotating from 0 to 360 degrees azimuth."
)]
