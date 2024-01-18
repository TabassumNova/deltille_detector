# converted from https://www.codeconvert.ai/app
# copilot



import os
import cv2
import numpy as np
from typing import List
from pathlib import Path
from argparse import ArgumentParser

class CalibrationCorner:
    def __init__(self, x: float, y: float, z: float, is_ordered: bool):
        self.x = x
        self.y = y
        self.z = z
        self.isOrdered = is_ordered

    def isValid(self):
        return True

def writeCornersToFile(os, corners: List[CalibrationCorner], filename: str, image_size: tuple, write_ordered_only: bool = True):
    num_corners = len(corners)
    if write_ordered_only:
        num_corners = sum(c.isValid() and c.isOrdered for c in corners)
    print(f"Writing {num_corners} corners to : {filename}")
    os.write(f"filename: {filename}\n")
    os.write(f"width: {image_size[0]}\n")
    os.write(f"height: {image_size[1]}\n")
    os.write(f"num_corners: {num_corners}\n")
    os.write("encoding: ascii\n")
    p = os.precision()
    os.precision(np.finfo(float).max_digits10)
    for c in corners:
        if not c.isValid() or (write_ordered_only and not c.isOrdered):
            continue
        os.write(f"{c.x} {c.y} {c.z}\n")
    os.precision(p)
    return os.good()

class DataSource:
    def __init__(self):
        self._last_file_name = ""

    def getImage(self, image, index = -1):
        if self.get_image(image, index):
            self.convert_to_grayscale(image)
            self.convert_type(image)
            return True
        else:
            return False

    def getLastFilename(self):
        return self._last_file_name

    def get_image(self, image, index):
        raise NotImplementedError

    def convert_to_grayscale(self, image):
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    def convert_type(self, image):
        if image.dtype == np.uint16:
            max_val = np.max(image)
            image = (image / max_val * 255).astype(np.float32)

class ImageListDataSource(DataSource):
    def __init__(self, filenames: List[str]):
        super().__init__()
        self._counter = 0
        self._filenames = filenames

    def get_image(self, I, f_i):
        if f_i < 0:
            f_i = self._counter
        if f_i < len(self._filenames):
            self._last_file_name = self._filenames[f_i]
            I = cv2.imread(self._filenames[f_i], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
            return not I is None
        else:
            return False

class TargetDetector:
    def __init__(self, target_dsc_fn):
        pass

    def run(self, I, corners, output_image):
        pass

def RunDetector(data_source: DataSource, target_dsc_fn: str, output_dir: str, save_images: bool):
    target_detector = TargetDetector(target_dsc_fn)
    I = np.zeros((1, 1))
    output_image = np.zeros((1, 1))
    for i in range(0, float('inf')):
        if data_source.getImage(I, i):
            if not I is None:
                corners = []
                target_detector.run(I, corners, output_image if save_images else None)
                filename = data_source.getLastFilename()
                filepath = Path(filename)
                outpath = Path(output_dir) if output_dir else filepath.parent
                basename = filepath.stem
                out_orpc_fn = outpath / (basename + ".orpc")
                with open(out_orpc_fn, "w") as fo:
                    writeCornersToFile(fo, corners, filename, I.shape, True)
                if save_images:
                    out_basename = "out_" + basename
                    out_img_fn = outpath / (out_basename + ".png")
                    print(f"Writing detection image to : {out_img_fn}")
                    cv2.imwrite(out_img_fn, output_image)

def main():
    parser = ArgumentParser()
    parser.add_argument("-t", "--target", dest="target_dsc_fn", required=True, help="Target *.dsc file")
    parser.add_argument("-f", "--files", dest="files", nargs="+", help="List of image files")
    parser.add_argument("-o", "--output", dest="output_dir", help="Output directory")
    parser.add_argument("-s", "--save-images", dest="save_images", action="store_true", help="Store debug images")
    args = parser.parse_args()

    target_dsc_fn = args.target_dsc_fn
    output_dir = args.output_dir if args.output_dir else ""
    files = args.files if args.files else []

    if not os.path.exists(target_dsc_fn) or not os.path.isfile(target_dsc_fn):
        raise ValueError(f"invalid target *.dsc file '{target_dsc_fn}'")

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        elif not os.path.isdir(output_dir):
            raise ValueError("argument to --output is NOT a directory")

    if files:
        data_source = ImageListDataSource(files)
        RunDetector(data_source, target_dsc_fn, output_dir, args.save_images)

    print("end")

if __name__ == "__main__":
    main()


