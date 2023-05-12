
<!-- chú ý đọc rõ code: khi train sẽ tạo ra file .pth mỗi _C.TRAIN.SAVE_FREQUENCY lần
    và ở epoch cuối cùng -->

<!-- Khi True thì sẽ print ra ở google collab: dòng 367 nằm trong loss/metric.py-->
_C.TRAIN.GOOGLE_COLAB = False

<!-- True thì train from checkpoint: dòng 163 train.py -->
_C.TRAIN.TRAIN_CHECKPOINT = False

<!-- Đường dẫn của file pretrain lưu thông tin của optimizer, model, epoch, loss -->
_C.TRAIN.PRETRAIN_PATH = 'log/23_03_26-resnet-tempr4-video_per:0.6-num_samplers:4-optimize:adam-loss:crossentropyloss/resnet-tempr4-video_per:0.6-num_samplers:4-optimize:adam-loss:crossentropyloss-epoch:0.pth'

<!-- Lưu đường dẫn của file config.yaml được tạo ra cho mỗi file log khi train, 
    chứa config của model, dùng để lưu tất cả thông tin và merge với file config: ở dòng 144 train.py -->
_C.TRAIN.PRETRAIN_CONFIG = ''

<!-- Nhớ đổi thuộc tính sau torng file config.yaml-->
TRAIN.PRETRAIN_PATH thành 