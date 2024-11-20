#!/usr/bin/env python
# coding: utf-8

import logging
from io import BytesIO
from typing import List, Dict, Tuple, Optional

import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from sklearn.cluster import KMeans
from lavis.models import load_model_and_preprocess

from constant import Constant
from image2tag import get_transform
from image2tag.models import image2tag

class ImageProcessor:
    def __init__(self):
        """Initialize any required models or resources."""
        # Configure logging
        logging.basicConfig(
            filename='image_processor.log',  # Log file path
            level=logging.INFO,  # Log level (INFO, DEBUG, ERROR, etc.)
            format="%(asctime)s - %(levelname)s - %(message)s",  # Log message format
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.face_recognition_model = YOLO(Constant.FACE_RECOGNITION_MODEL_PATH)
        self.image2tag_model = image2tag(
            pretrained=Constant.IMAGE2TAG_MODEL_PATH,
            image_size=Constant.IMAGE2TAG_SIZE,
            vit=Constant.VISION_TRANSFORMER_TYPE
        )
        self.image2tag_model.eval()

        self.image2tag_model = self.image2tag_model.to(self.device)
        self.transform = get_transform(image_size=Constant.IMAGE2TAG_SIZE)

        self.feature_exactrion_model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
            name=Constant.FEATURE_EXTRACTION_MODEL_NAME, 
            model_type="pretrain", 
            is_eval=True, device=self.device)
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        logging.info("ImageProcessor initialized.")
        
    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """
        Preprocess the image data for further processing.
        
        Convert to RGB if the image is not in RGB mode
        Resize to max dimension of 800px while maintaining aspect ratio
            
        Parameters:
        ----------
            image_data (bytes): Raw image data
            
        Returns:
        --------
            np.ndarray: Preprocessed image array or None if processing fails
            
        Examples
        --------
        >>> model = ImageProcessor()
        >>> with open("images/demo/group_persons.jpg", "rb") as f:
                raw_data = f.read()
        >>> processed_image = model.preprocess_image(raw_data)
        """
        try:
            # Open the image from raw bytes
            image = Image.open(BytesIO(image_data))
            
            # Convert to RGB if the image is not in RGB mode
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Resize the image while maintaining aspect ratio
            
            image.thumbnail((Constant.MAX_DIMENSION, Constant.MAX_DIMENSION))
            # Convert the image to a NumPy array
            
            image_array = np.array(image)
            logging.info(f"Image successfully preprocessed, size: {image_array.shape}")
            return image_array
    
        except Exception as e:
            logging.error(f"Error in preprocess_image: {e}")
            return None
            
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, any]]:
        """
        Detect faces in the image and return their locations and features.
        
        Parameters:
        ----------
            image (np.ndarray): Preprocessed image array
            
        Returns:
        --------
            List[Dict]: List of detected faces with their properties
                Each dict contain:
                - 'bbox': (x, y, width, height)
                - 'confidence': detection confidence
                - 'landmarks': facial landmarks, that is stores the coordinates of the 
                    left eye, right eye, nose, left edge, right edge, and the confidence points.
                
        Examples
        --------
        >>> model = ImageProcessor()
        >>> with open("images/demo/group_persons.jpg", "rb") as f:
                raw_data = f.read()
        >>> processed_image = model.preprocess_image(raw_data)
        >>> faces = img.detect_faces(processed_image)
        [
            {
                'bbox': (648.0, 20.0, 687.0, 69.0),
                'confidence': 0.8035881,
                'landmarks': {
                    'left_eye': array([ 657.59, 37.413, 0.93664], dtype=float32),
                    'right_eye': array([ 677, 38.899, 0.929], dtype=float32),
                    'nose': array([ 665.95, 48.852, 0.9235], dtype=float32),
                    'left_mouth': array([657.37, 55.657, 0.91659], dtype=float32),
                    'right_mouth': array([673.16, 56.872, 0.90834], dtype=float32)
                }
            },
            ...,
            {
                'bbox': (293.0, 109.0, 325.0, 151.0),
                'confidence': 0.8012416,
                'landmarks': {
                    'left_eye': array([300.31, 123.6, 0.90436], dtype=float32),
                    'right_eye': array([316.83, 123.18, 0.91103], dtype=float32),
                    'nose': array([309.18, 132.29, 0.90202], dtype=float32),
                    'left_mouth': array([302.43, 139.08, 0.89229], dtype=float32),
                    'right_mouth': array([315.93, 138.77, 0.89877], dtype=float32)
                }
            }
        ]
        """
        faces = []
        try:
            results = self.face_recognition_model.predict(image)
            for result in results:
                for box, keypoint in zip(result.boxes.data, result.keypoints.data):  # Access bounding box data
                    x1, y1, x2, y2, score, _ = box.cpu().numpy()
                    le, re, nose, lm, rm = keypoint.cpu().numpy()
                    faces.append({
                        "bbox":(x1, y1, x2, y2),
                        "confidence":score,
                        "landmarks":{
                            "left_eye": le,
                            "right_eye": re, 
                            "nose": nose, 
                            "left_mouth": lm, 
                            "right_mouth": rm
                        }
                    })
            logging.info(f"Detected {len(faces)} faces.")
            return faces, results
            
        except Exception as e:
            logging.error(f"Error in detect_faces: {e}")
            return faces
            
    def generate_tags(self, image: np.ndarray) -> List[Tuple[str, float]]:
        """
        Generate relevant tags for the image content with confidence scores.
        
        Parameters:
        ----------
            image (np.ndarray): Preprocessed image array
            
        Returns:
        --------
            List[Tuple[str, float]]: List of (tag, confidence) tuples

        Examples
        --------
        >>> model = ImageProcessor()
        >>> with open("images/demo/test_image.jpg", "rb") as f:
                raw_data = f.read()
        >>> processed_image = model.preprocess_image(raw_data)
        >>> model.generate_tags(processed_image)
        [('beret', 0.65494186),
         ('face', 0.83001745),
         ('flat', 0.9309285),
         ('icon', 0.9538902),
         ('illustration', 0.9483383),
         ('man', 0.999642),
         ('rectangle', 0.7039226),
         ('smile', 0.8700144),
         ('square', 0.90810025)]
         
        """
        list_tags = []
        try:
            image = self.transform(Image.fromarray(image.astype("uint8"), "RGB")).unsqueeze(0).to(self.device)
            with torch.no_grad():
                tags, scores = self.image2tag_model.generate_tag(image, threshold=Constant.IMAGE2TAG_THRESHOLD_CONFIDENCE)
            for tag, score in zip(list(tags), scores):
                list_tags.append((tag, score))
            while len(list_tags) < 5:
                list_tags.append(Constant.NONE_TAGS[0])
            logging.info(f"Generated {len(list_tags)} tags.")
            return list_tags
        except Exception as e:
            logging.error(f"Error in generate_tags: {e}")
            return Constant.NONE_TAGS*5

    def calculate_image_similarity(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """
        Calculate similarity score between two images.
        Returns a float value between 0 and 1 representing the similarity score between the two images.
        If any exception occurs, it safely returns 0.
        
        Parameters:
        ----------
            image1 (np.ndarray): First preprocessed image
            image2 (np.ndarray): Second preprocessed image
            
        Returns:
        --------
            float: Similarity score between 0 and 1
                
        Examples:
        --------
        >>> model = ImageProcessor()
        >>> with open("images/demo/biden2.jpg", "rb") as f:
                raw_data = f.read()
        >>> processed_image1 = model.preprocess_image(raw_data)
                
        >>> with open("images/demo/trump3.jpg", "rb") as f:
                raw_data = f.read()
        >>> processed_image2 = model.preprocess_image(raw_data)
        >>>  model.calculate_image_similarity(processed_image1, processed_image2)
        0.5553746223449707
        >>>  model.calculate_image_similarity(processed_image2, processed_image1)
        0.5553746223449707

        >>> with open("images/demo/trump2.jpg", "rb") as f:
                raw_data = f.read()
        >>> processed_image1 = model.preprocess_image(raw_data)
                
        >>> with open("images/demo/trump3.jpg", "rb") as f:
                raw_data = f.read()
        >>> processed_image2 = model.preprocess_image(raw_data)
        >>>  model.calculate_image_similarity(processed_image1, processed_image2)
        0.7459446787834167
        >>>  model.calculate_image_similarity(processed_image2, processed_image1)
        0.5553746223449707
        """
        try:
           
            feature_image1_sample = {
                "image": self.vis_processors["eval"](Image.fromarray(image1.astype("uint8"), "RGB")).unsqueeze(0).to(self.device),
                "text_input": self.txt_processors["eval"]("")
            }
            feature_image2_sample = {
                "image": self.vis_processors["eval"](Image.fromarray(image2.astype("uint8"), "RGB")).unsqueeze(0).to(self.device),
                "text_input": self.txt_processors["eval"]("")
            }
            features_image1 = self.feature_exactrion_model.extract_features(feature_image1_sample, mode="image")
            features_image2 = self.feature_exactrion_model.extract_features(feature_image2_sample, mode="image")
            
            similarity = self.cosine_similarity(features_image1.image_embeds_proj[:,0,:], features_image2.image_embeds_proj[:,0,:]).item()
            logging.info(f"Image similarity calculated: {similarity}")
            return round(similarity, 5)
            
        except Exception as e:
            logging.error(f"Error in calculate_image_similarity: {e}")
            return 0
    
    def extract_metadata(self, image_data: bytes) -> Dict[str, any]:
        """
        Extract useful metadata from the image.
        
        Parameters:
        ----------
            image_data (bytes): Raw image data
            
        Returns:
            Dict: Metadata dictionary containing:
                - dimensions (width, height)
                - color_space
                - format
                - size_bytes
                - dominant_colors (list of RGB values)
        Examples:
        --------
        >>> model = ImageProcessor()
        >>> with open("images/demo/biden2.jpg", "rb") as f:
                raw_data = f.read()     
        >>> model.extract_metadata(raw_data)
        {
            'dimensions': (1200, 675),
            'color_space': 'RGB', 'format': 'JPEG',
            'size_bytes': 61063,
            'dominant_colors': array([[131, 113,  88],
           [203, 181, 145],
           [ 59,  52,  44]])
        }
        """
        try:
            # Open the image from raw bytes
            image = Image.open(BytesIO(image_data))
            color_space = image.mode
            width, height = image.size
            image_format = image.format
            
            buffer = BytesIO()
            image.save(buffer, format=image.format)  # Use the image's original format
            size_bytes = len(buffer.getbuffer())  # or buffer.tell()
            
            image_array = np.array(image)
            dominant_colors = self._get_dominant_colors(image_array)
            logging.info(f"Metadata extracted: {metadata}")
            return {
                "dimensions":(width, height),
                "color_space":color_space,
                "format":image_format,
                "size_bytes":size_bytes,
                "dominant_colors":dominant_colors
            }
        except Exception as e:
            logging.error(f"Error in extract_metadata: {e}")
            return {}

    def _get_dominant_colors(self, image: np.ndarray) -> List[list]:
        """
        get dominant colors from image
        
        Parameters:
        ----------
            image (np.ndarray): Preprocessed image array
            
        Returns:
        --------
            list: list of RGB values
        """
        try:
            pixels = np.array(image).reshape(-1, 3)

            # Apply K-Means clustering
            kmeans = KMeans(n_clusters=Constant.DOMINANT_COLORS_CLUTES)
            kmeans.fit(pixels)
            
            # Get the cluster centers (dominant colors)
            dominant_colors = kmeans.cluster_centers_.astype(int)
            logging.info(f"Dominant colors extracted: {dominant_colors}")
            return dominant_colors
        except Exception as e:
            logging.error(f"Error in _get_dominant_colors: {e}")
            return []
        




