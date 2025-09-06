import requests
from typing import BinaryIO
from typing import Dict, Any, Optional

from dataclasses import dataclass

@dataclass
class ImageFeatures:
    x: float
    y: float
    cid: int
    local_x: float
    local_y: float
    local_z: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'x': self.x,
            'y': self.y,
            'cid': self.cid,
            'local_x': self.local_x,
            'local_y': self.local_y,
            'local_z': self.local_z
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ImageFeatures':
        return ImageFeatures(
            x=data['x'],
            y=data['y'],
            cid=data['cid'],
            local_x=data['local_x'],
            local_y=data['local_y'],
            local_z=data['local_z']
        )

@dataclass
class CharucoBoardConfig:
    squares_x: int
    squares_y: int
    square_length: float
    marker_length: float
    dictionary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'squares_x': self.squares_x,
            'squares_y': self.squares_y,
            'square_length': self.square_length,
            'marker_length': self.marker_length,
            'dictionary': self.dictionary
        }

def upload_image(file: BinaryIO, image_store_service_url: str) -> str:
    """
    Upload an image file to the image store service.

    Args:
        file: Binary file object (e.g., from open('image.jpg', 'rb'))
        image_store_service_url: Base URL of the image store service

    Returns:
        str: The image ID returned by the service

    Raises:
        requests.HTTPError: If the upload fails
    """
    files = {'file': file}

    response = requests.post(
        f"{image_store_service_url}/images",
        files=files
    )

    if not response.ok:
        raise requests.HTTPError(
            f"Failed to upload image: {response.status_code} {response.reason}"
        )

    data = response.json()
    return data['image_id']

def request_feature_detection(
    image_id: str,
    feature_detection_service_url: str,
    params: CharucoBoardConfig,
    return_overlay: bool = False
) -> list[ImageFeatures]:
    """
    Request feature detection on an uploaded image.

    Args:
        image_id: ID of the uploaded image
        pattern: Pattern to detect in the image
        feature_detection_service_url: Base URL of the feature detection service
        params: Additional parameters for detection (optional)
        return_overlay: Whether to return overlay data (default: False)

    Returns:
        Any: The detection results returned by the service

    Raises:
        requests.HTTPError: If the feature detection request fails
    """
    payload = {
        'image_id': image_id,
        'pattern': 'charuco',
        'params': params.to_dict(),
        'return_overlay': return_overlay
    }

    response = requests.post(
        f"{feature_detection_service_url}/detect_pattern",
        headers={'Content-Type': 'application/json'},
        json=payload
    )

    if not response.ok:
        raise requests.HTTPError(
            f"Feature detection failed: {response.status_code} {response.reason}"
        )

    response_data = response.json()
    # Extract the points from the response
    points = response_data.get('points', [])
    
    # Convert the points to ImageFeatures objects
    features = []
    for point in points:
        # Map the response field names to ImageFeatures field names
        feature = ImageFeatures(
            x=point['x'],
            y=point['y'],
            cid=point['id'],  # 'id' in response maps to 'cid' in ImageFeatures
            local_x=point['local_x'],
            local_y=point['local_y'],
            local_z=point['local_z']
        )
        features.append(feature)
    
    return features

# Example usage:
# result = request_feature_detection(
#     image_id="abc123",
#     pattern="circle",
#     feature_detection_service_url="http://localhost:8001",
#     params={"threshold": 0.8},
#     return_overlay=True
# )
