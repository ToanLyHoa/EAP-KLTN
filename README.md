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
```

Nhãn và đường dẫn file test đã có sẵn ở folder label/, train.csv và test.csv chính là tập split 01 của tập dữ liệu UCF-101

# Các loại mô hình:

* Nếu là mô hình chỉ dùng ResNet3D, run code với file train_reset3d.py

* Nếu mô hình có sự kết hợp giữa ResNet3D và Transformer, dùng file train_transformer.py

# 2. Hướng dẫn thực nghiệm mô hình

Chọn một file trọng số đã train ứng với file config .yaml, trong file .yaml. Điền đường dẫn file config vào PRETRAIN_CONFIG, đường dẫn file trọng số vào PRETRAIN_PATH. Sau đó kết quả thực nghiệm sẽ xuất hiện ở RESULT_DIR

```yaml
    DATA:
        EVAL_ONLY: true
    TRAIN: 
        PRETRAIN_CONFIG: ''
        PRETRAIN_PATH: ''
        RESULT_DIR: ''
```

Sau đó chỉnh config file trong file .py của mô hình tương ứng. Ví dụ, ta muốn eval mô hình resnet 3D với cách lấy mẫu ngẫu nhiên 16 frame, trong 30% quan sát:

Trong file train_resnet3d.py

```python
173    config_file = 'config/even_crop_random_resnet3d_18_sampler:1_per:0.3.yaml'
```
Trong file config tương ứng

```yaml
DATA:
    EVAL_ONLY: true

TRAIN:
    PRETRAIN_CONFIG: 'log/23_06_13-even_crop_random:1-resnet3d_50--video_per:0.3-video_len16-optimize:SGD-loss:crossentropylossmean/config.yaml'
    PRETRAIN_PATH: 'log/23_06_13-even_crop_random:1-resnet3d_50--video_per:0.3-video_len16-optimize:SGD-loss:crossentropylossmean/model_best.pth'
    RESULT_DIR: 'result'
```

# 3. Hướng dẫn huấn luyện mô hình

Với các file config có sẵn bao gồm thông tin của mạng backbone:

    - ResNet3D-18 và ResNet3D-50

    - Tên của các phương pháp lấy mẫu: 

        + normal_skip: là lấy mẫu rời rạc với skip cố định

        + auto_skip: là lấy mẫu rời rạc với skip ngẫu nhiên

        + ... có thể vào data_loader/video_sampler.py để biết rõ hơn về các phương pháp lấy mẫu dữ liệu này

Tương tự với khi thực nghiệm cho mô hình, ở đây ta chỉ cần thay thế vào file .py tương ứng train_resnet3d.py hoặc train_transformer.py config file tương ứng

```python
173    config_file = 'config/even_crop_random_resnet3d_18_sampler:1_per:0.3.yaml'
```
