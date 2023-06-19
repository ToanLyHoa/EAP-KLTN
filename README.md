# Hướng dẫn từng bước chạy mô hình

# 1. Công tác dữ liệu:

Tải tập dữ liệu UCF-101 [tại đây](https://www.crcv.ucf.edu/data/UCF101.php)

Sau đó tiền xử lí dữ liệu cho UCF-101 như ở [ResNet3D Pytorch](https://github.com/kenshohara/3D-ResNets-PyTorch#ucf-101)

Đặt tên cho folder dữ liệu là UCF-101-JPG sau đó chạy đoạn code sau để tạo file n_frame cho mỗi dãy ảnh

```python
    import os
    import glob

    def num_images(dataset_path,dst_path):

        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        for image_file in glob.glob(dataset_path+'/*/*/'):
            frames_count =0
            for files in os.listdir(image_file):
                if files.endswith('.jpg'):
                    frames_count +=1

            if frames_count<=0:
                print('{} does not have any frames'.format(image_file))
                continue
            
            count_frame_path = os.path.join(dst_path,image_file.split('/',maxsplit=1)[-1])
            if not os.path.exists(count_frame_path):
                os.makedirs(count_frame_path)

            with open(os.path.join(count_frame_path,'n_frames'),'w') as dst_file:
                dst_file.write(str(frames_count))

    num_images("UCF-101-JPG","UCF-101-JPG")

