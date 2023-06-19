'''
---  I M P O R T  S T A T E M E N T S  ---
'''
import math
import numpy as np
import coloredlogs, logging
import random
import cv2
coloredlogs.install()


'''
    [About]
        Randomly samples frames from the entire video without considering the order of frames. 
    [Init Args]
        - num: Integer for number of frames.
        - interval: Interger for the sampling interval. Defaults to 1.
        - speed: List or Tuple for the sampling speed based on the uniform distribution.
        - seed: Integer for randomisation.
    [Methods]
        - __init__ : Class initialiser, takes as argument the video path string.
        - set_num: Setter function for number of frames.
        - sampling: Main function for sampling.
'''
class RandomSampling(object):
    def __init__(self, num, interval=1, speed=[1.0, 1.0], seed=0):
        num = int(num)
        assert num > 0, "at least sampling 1 frame"
        self.frames = num
        self.num = num
        self.interval = interval if type(interval) == list else [interval]
        self.speed = speed
        self.rng = np.random.RandomState(seed)

    #setter for number of frames
    def set_num(self,new_num):
        assert new_num > 0, "at least sampling 1 frame"
        self.num = new_num

    # Setter for new number of sampled frames
    def set_num(self, new_num):
        self.num = new_num

    def sampling(self, range_max, v_id=None, prev_failed=False):
        if range_max < 1:
            range_max = 1
        interval = self.rng.choice(self.interval)
        if self.num == 1:
            return [self.rng.choice(range(0, range_max))]
        # sampling
        speed_min = self.speed[0]
        speed_max = min(self.speed[1], (range_max-1)/((self.num-1)*interval))
        if speed_max < speed_min:
            idxs = [self.rng.choice(range(0, range_max)) for _ in range(self.num)]
            idxs.sort()
            return idxs

        random_interval = self.rng.uniform(speed_min, speed_max) * interval

        frame_range = (self.num-1) * random_interval
        clip_start = self.rng.uniform(0, (range_max-1) - frame_range)
        clip_end = clip_start + frame_range
        idxs = np.linspace(clip_start, clip_end, self.num).astype(dtype=np.int32).tolist()
        return idxs


'''
    [About]
        Samples frames sequentially, with a fixed interval between each frame.     
    [Init Args]
        - num: Integer for number of frames.
        - interval: Interger for the sampling interval. Defaults to 1.
        - shuffle: Boolean for shuffling clips.
        - fix_cursor: Boolean for having a fixed cursor during sampling.
        - seed: Integer for randomisation.
    [Methods]
        - __init__ : Class initialiser, takes as argument the video path string.
        - set_num: Setter function for number of frames.
        - sampling: Main function for sampling.
'''
class SequentialSampling(object):
    def __init__(self, num, interval=1, shuffle=False, fix_cursor=False, seed=0):
        self.memory = {}
        self.num = int(num)
        self.interval = interval if type(interval) == list else [interval]
        self.shuffle = shuffle
        self.fix_cursor = fix_cursor
        self.rng = np.random.RandomState(seed)

    #setter for number of frames
    def set_num(self,new_num):
        self.num = int(new_num)

    def sampling(self, range_max, v_id, prev_failed=False):
        if range_max < 1:
            range_max = 1
        num = self.num
        interval = self.rng.choice(self.interval)
        frame_range = (num - 1) * interval + 1
        # sampling clips
        if v_id not in self.memory:
            clips = list(range(0, range_max-(frame_range-1), frame_range))
            if self.shuffle:
                self.rng.shuffle(clips)
            self.memory[v_id] = [-1, clips]
        # pickup a clip
        cursor, clips = self.memory[v_id]
        if not clips:
            idxs = [self.rng.choice(range(0, range_max)) for _ in range(self.num)]
            idxs.sort()
            return idxs
        cursor = (cursor + 1) % len(clips)
        if prev_failed or not self.fix_cursor:
            self.memory[v_id][0] = cursor
        # sampling within clip
        idxs = range(clips[cursor], clips[cursor]+frame_range, interval)
        return idxs

class RandomSequence(object):
    def __init__(self, num, **kwags):
        self.num = num

    def sampling(self, range_max, v_id, prev_failed=False):
        if self.num > range_max:
            print("The number of frames must be less than rangemax")
            return 
        start_index = random.randint(0, range_max - self.num + 1 )
        return [i for i in range(start_index, start_index + self.num)]

class RandomSequenceFromPoint(object):
    def __init__(self, num , **kwags):
        self.num = num

    def sampling(self, range_max, s, v_id, prev_failed = False):
        if range_max < self.num:
            range_max = self.num
        
        start = int(range_max/s)*(s-1)
        end = range_max

        if end - start < 16:
            start = end - 16

        start_index = random.randint(start, end - self.num + 1 )
        return [i for i in range(start_index, start_index + self.num)]
    
class NormalSampler(object):
    def __init__(self, video_per, frame_skip = 4 , **kwags):
        self.video_per = video_per
        self.frame_skip = frame_skip
        self.lenght_video = 187

    def sampling(self, video_len):
        
        start_index = 0

        end_index = int(self.lenght_video*self.video_per)

        offset = 0
        if end_index > video_len:
            offset = end_index - video_len

            indices = [i for i in range(start_index, video_len)]
        
            indices += [video_len - 1 for i in range(0, offset)]
        else:
            indices = [i for i in range(start_index, end_index)]

        indices = [indices[i] for i in range(0, len(indices), self.frame_skip + 1)]
        return indices

class SequenceSampler(object):
    def __init__(self, len_scale, frame_skip = 4, **kwags):

        assert  len_scale < frame_skip, "len_scale should be smaller than frame_skip"

        self.len_scale = len_scale
        self.frame_skip = frame_skip
        self.lenght_video = 187

    def sampling(self, video_len):
        
        start_index = 0

        end_index = int(self.lenght_video)

        offset = 0
        if end_index > video_len:
            offset = end_index - video_len

            indices = [i for i in range(start_index, video_len)]
        
            indices += [video_len - 1 for i in range(0, offset)]
        else:
            indices = [i for i in range(start_index, end_index)]

        # indices = [indices[i] for i in range(0, len(indices), self.frame_skip + 1)]
        # return indices
        indices_list = []
        for i in range(0, len(indices), self.frame_skip + self.len_scale):
            indices = []
            for j in range(self.len_scale):
                indices.append(i + j)
            indices_list.append(indices)

        return indices_list

class FixLengthSampler(object):
    def __init__(self, lenght = 0, skip = 0):
        
        self.lenght = lenght
        self.skip = skip
        
        pass

    def sampling(self, video_len = None):
        indices_list = [i*(self.skip + 1) if i*(self.skip + 1) < (video_len - 1) 
                                            else (video_len - 1)
                        for i in range(0, self.lenght)]
        return indices_list

class Auto_skip(object):
    def __init__(self, num_frame, percentage = 0.5):

        self.num_frame = num_frame
        self.per = percentage

    def sampling(self, frame_count = 187):

        n_frames = int(frame_count*self.per)
        if n_frames >= self.num_frame:
            frames = random.sample(range(0, n_frames), self.num_frame)
            frames.sort()
        else:
            frames = [i for i in range(0, n_frames)]
            for i in range(n_frames, self.num_frame):
                frames.append(n_frames - 1)
        
        return frames

# class Auto_skip(object):
#     def __init__(self,num_frame , percentage = 0.5):
#         self.num_frame = num_frame
#         self.per = percentage

#     def sampling(self,frame_count = 187):

#         n_frames = int(frame_count*self.per)
#         range_max = self.num_frame
        
#         if n_frames > self.num_frame:
#             remainder = self.num_frame % n_frames
#             skip = n_frames // self.num_frame
#         else:
#             remainder = 0
#             skip = 1
#             range_max = n_frames
            
#         frames = [i*skip + random.randint(0,remainder) if remainder < skip
#                   else i * skip + random.randint(0,skip)
#             for i in range(0,range_max)]
        
#         while(len(frames)<self.num_frame):
#             frames.append(n_frames)
            
#         return frames
import math, random
class LoopPadding(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out

class TemporalRandomEvenCrop(object):

    def __init__(self, size, n_samples=1, percentage = .5):
        self.size = size
        self.n_samples = n_samples
        self.loop = LoopPadding(size)
        self.per = percentage

    def sampling(self, frame_count):


        n_frames = int(frame_count*self.per)
        frame_indices = [i for i in range(n_frames)]

        stride = 1
        if self.n_samples != 1:
            stride = max(
                1, math.ceil((n_frames - 1 - self.size) / (self.n_samples - 1)))

        out = []
        count = self.n_samples
        while count != 0:
            for begin_index in frame_indices[::stride]:
                count -= 1
                if len(out) >= self.n_samples:
                    break

                if stride != 1 and stride > self.size:
                    end_index = min(frame_indices[-1] + 1, begin_index + stride)
                else:
                    end_index = min(frame_indices[-1] + 1, begin_index + self.size)


                offset = end_index - begin_index
                if offset >= self.size:
                    sample = random.sample(range(begin_index, end_index), self.size)
                else:
                    sample = random.sample(range(begin_index, end_index), offset)

                sample.sort()

                if len(sample) < self.size:
                    out.append(self.loop(sample))
                    break
                else:
                    out.append(sample)

                if count == 0:
                    break

        return out

class TemporalEvenCrop(object):

    def __init__(self, size, n_samples=1, percentage = .5):
        self.size = size
        self.n_samples = n_samples
        self.loop = LoopPadding(size)
        self.per = percentage

    def sampling(self, frame_count):


        n_frames = int(frame_count*self.per)
        frame_indices = [i for i in range(n_frames)]

        stride = 1
        if self.n_samples != 1:
            stride = max(
                1, math.ceil((n_frames - 1 - self.size) / (self.n_samples - 1)))
        # stride = 1

        out = []
        count = self.n_samples
        while count != 0:
            for begin_index in frame_indices[::stride]:
                count -= 1
                if len(out) >= self.n_samples:
                    break

                if stride != 1 and stride > self.size:
                    end_index = min(frame_indices[-1] + 1, begin_index + stride)
                else:
                    end_index = min(frame_indices[-1] + 1, begin_index + self.size)


                offset = end_index - begin_index
                if offset >= self.size:
                    start = random.sample(range(begin_index, end_index - self.size + 1), 1)[0]
                    sample = list(range(start, start + self.size))
                else:
                    sample = random.sample(range(begin_index, end_index), offset)

                sample.sort()


                if len(sample) < self.size:
                    out.append(self.loop(sample))
                    break
                else:
                    out.append(sample)

                if count == 0:
                    break

        return out

class FrameDifference(object):
    def __init__(self,num,**kwagrs):
        self.num=num

    def sampling (self,s,video_path,num_frames,v_id, prev_failed = False ):
        '''
        [About]
            Take frames indices that are significantly different from previous frames.
        [Args]
            s: scale of video (eg: 1/2 ,1/3 ,..)\n
            video_path: path to avi video (eg: UCF-101-test/Archery/v_Archery_g01_c01)
            num_frames: the number of frames of video
        [Returns]
            frames: list array of frame indices 
        '''
        video_path = video_path+'.avi'
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = min(frame_count,num_frames)
        prev_frame = None
        frames = []
        k=0
        for i in range(int(frame_count*s)):
            ret, frame = cap.read()
            if not ret:
                break
            if prev_frame is None or cv2.absdiff(frame, prev_frame).mean() > 7*s:
                frames.append(i)
                k=i
                if len(frames) == self.num:
                    break
            prev_frame = frame
        # Kiểm tra nếu không đủ số frames thì lấy thêm từ frame cuối
        while len(frames) < self.num:
            frames.append(k)
        cap.release()
        return frames

if __name__ == "__main__":

    import logging
    logging.getLogger().setLevel(logging.DEBUG)

    """ test RandomSampling() """

    random_sampler = RandomSampling(num=8, interval=2, speed=[0.5, 2])

    logging.info("RandomSampling(): range_max (6) < num (8)")
    for i in range(10):
        logging.info("{:d}: {}".format(i, random_sampler.sampling(range_max=6, v_id=1)))

    logging.info("RandomSampling(): range_max (8) == num (8)")
    for i in range(10):
        logging.info("{:d}: {}".format(i, random_sampler.sampling(range_max=8, v_id=1)))

    logging.info("RandomSampling(): range_max (32) > num (8)")
    for i in range(10):
        logging.info("{:d}: {}".format(i, random_sampler.sampling(range_max=32, v_id=1)))


    """ test SequentialSampling() """
    sequential_sampler = SequentialSampling(num=8, interval=3, fix_cursor=False)

    logging.info("SequentialSampling(): range_max (6) < num (8)")
    for i in range(10):
        logging.info("{:d}: v_id = {}: {}".format(i, 0, list(sequential_sampler.sampling(range_max=3, v_id=0))))

    logging.info("SequentialSampling(): range_max (8) == num (8)")
    for i in range(10):
        logging.info("{:d}: v_id = {}: {}".format(i, 1, list(sequential_sampler.sampling(range_max=9, v_id=1))))

    logging.info("RandomSampling(): range_max (32) > num (8)")
    for i in range(10):
        logging.info("{:d}: v_id = {}: {}".format(i, 2, list(sequential_sampler.sampling(range_max=32, v_id=2))))
