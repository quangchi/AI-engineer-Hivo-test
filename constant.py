class Constant:
    FACE_RECOGNITION_MODEL_PATH = "model_trained/face_recognite_model.pt"
    IMAGE2TAG_MODEL_PATH = "model_trained/image2tag_model.pth"
    VISION_TRANSFORMER_TYPE = "swin_l"
    MAX_DIMENSION = 800
    IMAGE2TAG_SIZE = 384
    NONE_TAGS = [('none', 0)]
    IMAGE2TAG_THRESHOLD_CONFIDENCE = 0.5
    FEATURE_EXTRACTION_MODEL_NAME = "blip2_feature_extractor"
    FEATURE_EXTRACTION_SAMPLE = {"image": None, "text_input": ""}
    DOMINANT_COLORS_CLUTES = 3