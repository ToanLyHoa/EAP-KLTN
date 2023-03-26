import os
import csv
import cv2
import random
import numpy as np
import sqlite3
import torch
import torch.utils.data as data
import coloredlogs, logging
import torch.nn.functional as F
import linecache
import sys
import time
from datetime import datetime
import json
from PIL import Image

class Video(object):
    def __init__(self, video_path, video_transform=None, end_size=(16,244,244), percentage=1.):
        self.path=video_path
        self.video_path=os.path.join(video_path,'frames.db')
        self.frame_path=os.path.join(video_path,'n_frames')
        self.end_size=end_size
        self.video_per=percentage
        self.video_transform=video_transform

    def reset(self):
        self.video_path = None
        self.frame_path = None
        self.frame_count = -1
        return self
    
    # Đếm số lượng khung hình của một phần video
    def count_frames(self):
        if (os.path.isfile(self.frame_path)):
            self.frame_count = int(int(open(self.frame_path,'r').read()) * self.video_per)
        else:
            logging.error('Directory {} is empty!'.format(self.frame_path))
            raise Exception("Empty directory !")
        return self.frame_count

    # Trích xuất các khung hình từ sqlite 
    def extract_frames(self, indices):
        frames = []
        # Kết nối đến SQLite database
        con = sqlite3.connect(self.video_path)
        cur = con.cursor()

        # Lấy tên các khung hình dựa trên các chỉ số trong indices
        frame_names = ["{}/{}".format(self.path.split('/')[-1], 'frame_%05d' % (index+1)) for index in indices]

        # Tạo SQL query để lấy các khung hình tương ứng với tên khung hình
        sql = "SELECT Objid, frames FROM Images WHERE ObjId IN ({seq})".format(seq=','.join(['?']*len(frame_names)))
        row = cur.execute(sql, frame_names).fetchall()

        # Sắp xếp các khung hình theo thứ tự trong video và thêm vào list frames
        for ObjId, item in sorted(row, key=lambda pair: pair[0]):
            nparr = np.frombuffer(item, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)

        # Chuyển đổi list frames sang dạng numpy array
        frames = np.asarray(frames)


        # Áp dụng phép biến đổi video nếu có
        if self.video_transform is not None:
            # code cũ
            # frames = self.video_transform(frames, self.end_size)

            self.video_transform.randomize_parameters()
            frames = [self.video_transform(Image.fromarray(frame)) for frame in frames]
            frames = torch.stack(frames, 0).permute(1, 0, 2, 3)
            # Kiểm tra số lượng khung hình và thực hiện nội suy nếu số lượng khung hình không đúng
            if frames.shape[1] != self.end_size[0]:
                frames = F.interpolate(frames.unsqueeze(0), size=self.end_size, mode='trilinear',align_corners=False).squeeze(0)
        else:
            if frames.shape[0] != self.end_size[0]:
                frames = F.interpolate(frames.unsqueeze(0), size=self.end_size, mode='trilinear',align_corners=False).squeeze(0)
        
        # Đóng kết nối
        cur.close()
        con.close()
        return frames


class VideoIter(data.Dataset):
    
    def __init__(self, csv_filepath, sampler, video_transform, cfg):
        super(VideoIter, self).__init__()
        
        self.csv_filepath = csv_filepath
        self.sampler = sampler  
        self.video_transform = video_transform

        self.gpu = cfg.USE_GPU
        self.clip_size = cfg.CLIP_SIZE
        self.video_per = cfg.VIDEO_PER
        self.dataset_location = cfg.DATA_DIR
        self.train_file = cfg.TRAIN_FILE
        self.val_file = cfg.VAL_FILE
        self.return_video_path = cfg.RETURN_VIDEO_PATH
        assert cfg.NUM_SAMPLERS >= 1, 'VideoIter: The number of samplers cannot be smaller than 1!'
        self.num_samplers = cfg.NUM_SAMPLERS
        
        self.rng = np.random.RandomState(cfg.SHUFFLE_LIST_SEED if cfg.SHUFFLE_LIST_SEED else 0)
        # self.randomise = self.randomise    

        self.video_dict = self.get_video_dict(self.dataset_location, self.csv_filepath, self.video_per)
        # Create array to hold the video indices
        self.indices = list(self.video_dict.keys())

        # Shuffling indices array
        if cfg.SHUFFLE_LIST_SEED is not None:
            self.rng.shuffle(self.indices)
        logging.info("VideoIter:: iterator initialized (phase: '{:s}', num: {:d})".format(csv_filepath, len(self.indices)))

        # load dictionary label
        # self.label_dict = json.loads('UCF-101-testcsv/dictionary.json')

    def __len__(self):
        return len(self.indices)
    
    def getitem_array_from_video(self,index):
        v_id,label,vid_path,frame_count = self.video_dict.get(index)
        try:
            video = Video(video_path = vid_path, 
                          video_transform = self.video_transform, 
                          end_size = self.clip_size,
                          percentage = self.video_per)
            if frame_count < 0:
               frame_count = video.count_frames()
            
            sampled_frames = []
            for s in range(1, self.num_samplers+1):
                range_max = int(frame_count * (s/self.num_samplers))
                sampled_indices = self.sampler.sampling(range_max = range_max, s = s, v_id = v_id)
                sampled_frames.append(video.extract_frames(sampled_indices).unsqueeze(0))
            
            sampled_frames = torch.cat(sampled_frames,dim=0)

        except IOError as e:
            logging.warning(">> I/O error({0}): {1}".format(e.errno, e.strerror))
        
        return sampled_frames ,label, vid_path
   
    def __getitem__(self, index):
        succ = False
        while not succ:
            try:
                index = int(index)
                if (index == 0):
                    index += 1
                frames, label, vid_path = self.getitem_array_from_video(index)
                _, _, t, h, w = frames.size()
                if (t!=self.clip_size[0] and h!=self.clip_size[1] and w!=self.clip_size[2]):
                    raise Exception('Clip size should be ({},{},{}), got clip with: ({},{},{})'.format(*self.clip_size,t,h,w))
                succ = True
            except Exception as e:

                exc_type, exc_obj, tb = sys.exc_info()
                f = tb.tb_frame
                lineno = tb.tb_lineno
                filename = f.f_code.co_filename
                linecache.checkcache(filename)
                line = linecache.getline(filename, lineno, f.f_globals)
                message = 'Exception in ({}, line {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj)

                prev_index = index
                #index = self.rng.choice(range(0, self.__len__()))
                d_time = int(round(datetime.now().timestamp() * 1000)) # Ensure randomisation
                index = random.randrange(d_time % self.__len__())
               
        if self.return_video_path:
                return frames, label, vid_path
        else:
                return frames, label

    # Get video info (label,path,num_frame_partial) from file csv
    def get_video_dict(self, dataset_location, csv_filepath, video_per):

        assert os.path.exists(dataset_location), "VideoIter:: failed to locate dataset at given location: `{}'".format(dataset_location)
        assert os.path.exists(csv_filepath), "VideoIter:: failed to locate csv file at given location: `{}'".format(csv_filepath)
        
        found_videos=0
        videos_dict={}
        ids=[]

        # Store dictionary of labels keys:'str' , values:'int' to a .JSON file (as a common reference between dataset sets)
        if (self.train_file in csv_filepath):
            labels_dict_filepath = csv_filepath.split('train')[0]+'dictionary.json'
        elif (self.val_file in csv_filepath):
            labels_dict_filepath = csv_filepath.split('val')[0]+'dictionary.json'
        else:
            labels_dict_filepath = csv_filepath.split('test')[0]+'dictionary.json'

        if (os.path.exists(labels_dict_filepath)):

            with open(labels_dict_filepath) as json_dict:
                labels_dict = json.loads(json_dict.read())
            label_last_index = len(labels_dict)
        else:
            labels_dict = {}
            label_last_index = 0

        for i,line in enumerate(csv.DictReader(open(csv_filepath))):
            id=line.get('id').strip()
            ids.append(id)
            video_path=os.path.join(dataset_location,line.get('label'),id)

            # Check if label has already been found and if not add it to the dictionary:
            if not (line.get('label') in labels_dict):
                labels_dict[line.get('label')] = label_last_index
                label_last_index += 1

            assert os.path.exists(video_path), "VideoIter:: failed to locate csv file at given location: `{}'".format(video_path)
            found_videos+=1
       
            with open(os.path.join(video_path,'n_frames')) as f:
                frame_count = int(int(f.readline()) * video_per)

            info = [found_videos, line.get('label'), video_path, frame_count]
            videos_dict[found_videos] = info 


        # Convert to indicing in alphabetical order
        for j,key in enumerate(sorted(labels_dict.keys())):
            labels_dict[key] = j

        # Convert `videos_dict` labels to numeric
        for key,value in videos_dict.items():
            videos_dict[key][1] = labels_dict[videos_dict[key][1]]

        for k in sorted(labels_dict.keys()):
            num = 0
            for key,value in videos_dict.items():
                if (videos_dict[key][1] == labels_dict[k]):
                    num+=1

        # Save dictionary if it does not already exists
        if not (os.path.exists(labels_dict_filepath)):
            logging.info("VideoIter:: Dictionary saved at {} \n".format(labels_dict_filepath))
            with open(labels_dict_filepath,'w') as json_dict:
                json.dump(labels_dict,json_dict)
        else:
            logging.info("VideoIter:: Found dict at: {} \n".format(labels_dict_filepath))
        return videos_dict          


        




