import numpy as np
import torch
import imgaug.augmenters as iaa
'''
    [About]
        Function for composing together differengt transforms. If normalisation is enabled, it will also normalise
        the transformed videos at the end.
    [Init Args]
        - transforms: imgaug.Sequential object that contains all the transforms to be applied. Transforms will be the
        same for each frame in a video to ensure continuity.
        - normalise: Boolean for applying normalisation at the end of transformations. It will normally be the
        ImageNet mean and std. Defaults to None.
    [Methods]
        - __init__ : Class initialiser, takes as argument the video path string.
        - __call__ : Main class call function for applying transformations and performing normalisation if not None.
'''
class Compose(object):
    def __init__ (self,transforms = None, normalise = None):
        self.transforms = transforms
        self.normalise = normalise

    def __call__(self, data, end_size = (8, 224, 224)):
        # imgaug package requires batch size - apply same tranform to all frames
        T,H,W,C=data.shape
        # vid_aug  = self.transforms.to_deterministic()

        # Scale video to have shoter size of 384
        scaling = iaa.Resize({"shoter-side":384,"longer-side":"keep-aspect-ratio"})
        if H>=384 and W >= 384:
            # Resize video to take center crop of 384x384
            center_crop = iaa.CenterCropToFixedSize(width=384,height=384)
            center_crop_aug = iaa.Sequential([scaling, center_crop])
            center_crop_aug = center_crop_aug.to_deterministic()         
        else:
            center_crop_aug = iaa.Sequential([scaling])
            center_crop_aug = center_crop_aug.to_deterministic()                       
       
        # Padding to make sure the image size is at least 224x224
        padding = iaa.Sequential([
            iaa.PadToFixedSize(width=max(W,end_size[1]),height=max(H,end_size[2]))
        ])
        padding_aug=padding.to_deterministic()

        # Random crop of size 224x224
        random_crop = iaa.CropToFixedSize(width=end_size[1], height=end_size[2])
        random_crop_aug = random_crop.to_deterministic()

        # Apply image augmentations to all frames in the video

        if self.transforms != None:
            data_aug = self.transforms.augment_images(data)
        else:
            data_aug = data

        # Apply center crop and scaling to all frames in the video
        data_aug = [center_crop_aug.augment_image(frame) for frame in data_aug]   

        # Apply padding to all frames
        if data_aug[0].shape[0] < 224 or data_aug[0].shape[1] < 224:
            data_aug = [padding_aug.augment_image(frame) for frame in data_aug]

        # Apply random crop to all frames in the video
        data_aug = [random_crop_aug.augment_image(frame) for frame in data_aug]   
        
        new_data_shape=str(np.asarray(data_aug).shape)
        new_data = np.asarray(data_aug)

        # Convert video to Tensor
        if isinstance(new_data,np.ndarray):
            new_data = torch.from_numpy(new_data).permute((3,0,1,2)) # np.arr[batch,height,width,channel] => tensor[chanel,batch,height,width]
            new_data =  (new_data.float() / 255.0)

        # Check if normalisation is to be applied
        if self.normalise is not None:
            for d, m, s in zip(new_data, self.normalise[0], self.normalise[1]):
                d.sub_(m).div_(s) 
        return new_data

'''
    [About]
        Base class for all transformations.
    [Methods]
        - set_random_state: Function for randomisation based on seed.
'''
class Transform(object):
    def set_random_state(self, seed=None):
        self.rng = np.random.RandomState(seed)