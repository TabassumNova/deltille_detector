import os
import subprocess
import pandas as pd
import numpy as np
from structs.struct import struct


class Image_Parser:
    def __init__(self, app_path, dsc_path, camera_path, output_path) -> None:
        self.app_path = app_path
        self.dsc_path = dsc_path
        self.camera_path = camera_path
        self.output_path = output_path
        self.total_board = 20
        self.boards = {}
        self.image_dict = {} 

    def generate_deltille_output(self):
        for images in os.listdir(self.camera_path):
            if (images.endswith(".png") or images.endswith(".jpg") or images.endswith(".jpeg")):
                img_path = os.path.join(self.camera_path, images)
                output = subprocess.check_output([self.app_path,"-t", 
                                                  self.dsc_path, "-f", img_path, "-o", self.output_path, "-s"])
                pass
        pass

    def encode_dsc(self):
        # find number of boards -> self.total_board
        
        with open(self.dsc_path) as file:
            lines = file.readlines()
            for line in lines:                       
                cols = line[:-1].split(',')
                if len(cols) == 4:
                    board_id = int(cols[0])
                    size = [int(cols[1])+1, int(cols[2])+1]
                    edge_size = float(cols[3])
                    
                if len(cols) == 2:
                    tag_family = cols[0]
                    tag_border = float(cols[1])
                    
                if len(cols) == 6:
                    if board_id not in self.boards:
                        self.boards[board_id] = struct(size = size, edge_size = edge_size,
                                                  tag_border = tag_border, corners_3D = [])
                        
                    self.boards[board_id].corners_3D.append([float(cols[3]), float(cols[4]), 0.0])

        pass

    def encode_deltille_output(self):
        
        for filename in os.listdir(self.output_path):
            
            if filename.endswith(".txt"):
                img_name = filename.split('.')[0]
                self.image_dict[img_name] = []
                file_path = os.path.join(self.output_path, filename)
                board_df = pd.DataFrame(index=list(np.arange(0,self.total_board)))
                with open(file_path) as file:
                    lines = file.readlines()
                    board_ids = []
                    corner_ids = []
                    corners = []
                    for line in lines[5:]:                       
                        cols = line[:-2].split(',')
                        board_num = int(cols[0])
                        corner_num = int(cols[1])
                        corner_x = float(cols[3])
                        corner_y = float(cols[4])
                        board_ids.append(board_num)
                        corner_ids.append(corner_num)
                        corners.append([corner_x, corner_y])
                    for b in range(0, self.total_board):
                        index_list = []
                        if b in board_ids:
                            index_list = [idx for idx, i in enumerate(board_ids) if i == b]
                            self.image_dict[img_name].append(struct(corner_ids = [corner_ids[i] for i in index_list],
                                                        corners = [corners[i] for i in index_list]))
                        else:
                            self.image_dict[img_name].append(struct(corner_ids = [], corners = []))
        pass
        

if __name__ == "__main__":
    i = Image_Parser(app_path="./build/apps/deltille_detector", dsc_path="./test/input/ico_deltille.txt",
                     camera_path="./test/input", output_path="./test/output")
    i.encode_dsc()
    i.generate_deltille_output()
    i.encode_deltille_output()
    pass
