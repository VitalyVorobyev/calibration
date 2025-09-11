#! /usr/bin/env python
""" Use services to process images and write results to a file """

import os
import argparse
import json
import tqdm

from dataclasses import dataclass

from apis import (
    upload_image,
    request_feature_detection,
    requests,
    ImageFeatures,
    CharucoBoardConfig,
)

ISS_URL = os.environ.get("ISS_URL", "http://localhost:8000")
FDS_URL = os.environ.get("FDS_URL", "http://localhost:8080")

@dataclass
class ImageData:
    path: str
    features: list[ImageFeatures]

class ImageProcessor:
    def __init__(self, board_config: CharucoBoardConfig):
        self.iss_url = ISS_URL
        self.fds_url = FDS_URL
        self.board_config = board_config
        self.imgdata = {}

    def send_to_iss(self, imfile: str) -> bool:
        """ Send image to ISS and return the response """
        try:
            with open(imfile, 'rb') as f:
                image_id = upload_image(f, "http://localhost:8000")
                self.imgdata[image_id] = ImageData(path=imfile, features=[])
                return True
        except requests.HTTPError as e:
            print(f"Error uploading image {imfile}: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error uploading image {imfile}: {e}")
            return False

    def fetch_features(self) -> bool:
        """ Fetch features for the given image ID from FDS """
        try:
            for img_id, img_data in tqdm.tqdm(self.imgdata.items()):
                features = request_feature_detection(
                    image_id=img_id,
                    feature_detection_service_url=self.fds_url,
                    params=self.board_config,
                    return_overlay=False
                )
                print(f"Fetched {len(features)} features for image {img_data.path.split('/')[-1]}")
                img_data.features = features
            return True
        except requests.HTTPError as e:
            print(f"Error fetching features: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error fetching features: {e}")
            return False

def process_image_json(fname: str, out_path: str) -> list[ImageProcessor]:
    """Process images.json file and write detected features to ``out_path``.

    The resulting JSON structure is compatible with the calibration C++ app and
    looks like::

        {
            "cameras": [
                [  # camera 0 views
                    [ {"x": X, "y": Y, "u": U, "v": V}, ... ],
                    ...
                ],
                [  # camera 1 views
                    ...
                ]
            ]
        }

    Parameters
    ----------
    fname : str
        Path to ``images.json`` describing pairs of images for each camera.
    out_path : str
        Destination JSON file where extracted features will be written.
    """
    path = os.path.dirname(fname)
    board = CharucoBoardConfig(
        squares_x=22,
        squares_y=22,
        square_length=1.362,
        marker_length=1.362 * 0.75,
        dictionary="DICT_4X4_1000",
    )
    processors = [ImageProcessor(board) for _ in range(2)]
    with open(fname, "r") as f:
        data = json.load(f)
        for item in data:
            for proc, imgname in zip(processors, item):
                print(f"Processing {imgname}")
                proc.send_to_iss(f"{path}/{imgname}")

    for proc in processors:
        proc.fetch_features()

    # Serialize features to observations for calibration input
    cameras = []
    for proc in processors:
        cam_views = []
        for img_data in proc.imgdata.values():
            obs = [
                {
                    "x": feat.local_x,
                    "y": feat.local_y,
                    "u": feat.x,
                    "v": feat.y,
                }
                for feat in img_data.features
            ]
            cam_views.append(obs)
        cameras.append(cam_views)

    with open(out_path, "w") as f:
        json.dump({"cameras": cameras}, f, indent=2)

    return processors

def main():
    parser = argparse.ArgumentParser(
        description="Process images and write detected features to a JSON file"
    )
    parser.add_argument("-p", "--poses", help="poses.json file")
    parser.add_argument("-i", "--images", help="images.json file")
    parser.add_argument("-b", "--board", help="ChArUco board config file")
    parser.add_argument(
        "-o", "--output", default="features.json", help="Output JSON file"
    )
    args = parser.parse_args()

    if args.images:
        process_image_json(args.images, args.output)

if __name__ == "__main__":
    main()
