
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


'config/resnet3d_18_sampler:auto_skip:11_per:0.3.yaml',
'config/resnet3d_18_sampler:auto_skip:14_per:0.3.yaml',
'config/resnet3d_18_sampler:auto_skip:19_per:0.3.yaml',
'config/resnet3d_18_sampler:auto_skip:19_per:0.5.yaml',
'config/resnet3d_18_sampler:auto_skip:24_per:0.5.yaml',
'config/resnet3d_18_sampler:auto_skip:28_per:0.3.yaml',
'config/resnet3d_18_sampler:auto_skip:28_per:0.7.yaml',
'config/resnet3d_18_sampler:auto_skip:31_per:0.5.yaml',
'config/resnet3d_18_sampler:auto_skip:35_per:0.7.yaml',
'config/resnet3d_18_sampler:auto_skip:47_per:0.5.yaml',
'config/resnet3d_18_sampler:auto_skip:47_per:0.7.yaml',
'config/resnet3d_18_sampler:auto_skip:56_per:0.3.yaml',
'config/resnet3d_18_sampler:auto_skip:70_per:0.7.yaml',
'config/resnet3d_18_sampler:auto_skip:94_per:0.5.yaml',
'config/resnet3d_18_sampler:auto_skip:140_per:0.7.yaml',

'config/resnet3d_50_sampler:auto_skip:11_per:0.3.yaml',
'config/resnet3d_50_sampler:auto_skip:14_per:0.3.yaml',
'config/resnet3d_50_sampler:auto_skip:19_per:0.3.yaml',
'config/resnet3d_50_sampler:auto_skip:19_per:0.5.yaml',
'config/resnet3d_50_sampler:auto_skip:24_per:0.5.yaml',
'config/resnet3d_50_sampler:auto_skip:28_per:0.3.yaml',
'config/resnet3d_50_sampler:auto_skip:28_per:0.7.yaml',
'config/resnet3d_50_sampler:auto_skip:31_per:0.5.yaml',
'config/resnet3d_50_sampler:auto_skip:35_per:0.7.yaml',
'config/resnet3d_50_sampler:auto_skip:47_per:0.5.yaml',
'config/resnet3d_50_sampler:auto_skip:47_per:0.7.yaml',
'config/resnet3d_50_sampler:auto_skip:56_per:0.3.yaml',
'config/resnet3d_50_sampler:auto_skip:70_per:0.7.yaml',
'config/resnet3d_50_sampler:auto_skip:94_per:0.5.yaml',
'config/resnet3d_50_sampler:auto_skip:140_per:0.7.yaml',

# EAP-KLTN

Chinh dong 227, dong 36 video_iterator.py frame_count = int(int(f.readline()) * video_per) bo di nhan voi video_per

Chinh dong 64 video_iterator.py cho 

if len(frame_names) - len(row):
            [frames.append(img) for i in range(len(frame_names) - len(row))]




* Thực nghiệm mô hình qua 21 lớp hành động thuộc bộ dữ liệu UCF-101 như em bé bò, phóng lao, bắn cung, nhảy dù, nhảy cao,... .Việc dự đoán hành động sớm trên các lớp hành động này đem lại ý nghĩa thiết thực trong cuộc sống hàng ngày. Ví dụ trong hành động em bé bò, việc dự đoán hành động có thể giúp cho các bậc phụ huynh và nhân viên chăm sóc trẻ nhỏ hiểu rõ hơn về giai đoạn phát triển của trẻ nhỏ và giúp cho việc chăm sóc và giáo dục trẻ nhỏ trở nên dễ dàng hơn. Hay trong hành động bắn cung, việc dự đoán hành động có thể giúp cho các VĐV, huấn luyện viên cải thiện kỹ thuật và hiệu suất thi đấu của mình.

* Chạy hàm def sampling thuộc  class FrameDifference(object):
video_iter.py: Hàm  def getitem_array_from_video(self,index) thuộc class VideoIter (line 138)

sampled_indices = self.sampler.sampling(s/self.num_samplers,vid_path,frame_count,v_id)