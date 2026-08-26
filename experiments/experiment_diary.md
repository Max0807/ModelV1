# ModelV1 实验日记

从下一次正式训练开始自动更新。此前实验不回填，`--dry-run` 不记录。

## 实验索引

| 完成时间 | Run | 实验说明 | 最佳 Val EPE | 相比上一实验 | 最佳 Epoch |
|---|---|---|---:|---:|---:|
| 2026-07-23T17:11:37+08:00 | 20260723_164504 | V2.2：双眼6D几何加入PnP质量门控与可学习残差修正。 | 20.694 mm | — | 217 |
| 2026-07-23T18:25:55+08:00 | 20260723_175740 | V2.2：双眼6D几何加入PnP质量门控与可学习残差修正。 | 21.211 mm | +0.517 mm | 269 |
| 2026-07-23T21:41:08+08:00 | 20260723_211817 | Crop消融：V2.2关闭crop_cam分支，保留scene、双眼6D几何和质量门控。 | 22.290 mm | +1.079 mm | 289 |
| 2026-07-23T23:06:05+08:00 | 20260723_224136 | Scene消融：恢复crop_cam，将Scene从full25替换为物理TableFrame 7D。 | 20.954 mm | -1.336 mm | 260 |
| 2026-07-24T00:09:08+08:00 | 20260723_234608 | Scene消融：保持crop_cam和V2.2质量门控，将Scene替换为桌面Orientation 6D。 | 21.948 mm | +0.994 mm | 233 |
| 2026-07-24T10:22:37+08:00 | 20260724_095707 | Scene消融：保持crop_cam和V2.2质量门控，将Scene替换为桌面Orientation 6D。 | 22.114 mm | +0.166 mm | 244 |
| 2026-07-24T11:29:41+08:00 | 20260724_110737 | Scene消融：保持crop_cam和V2.2质量门控，将Scene替换为桌面Orientation 6D。 | 20.783 mm | -1.331 mm | 248 |
| 2026-07-25T12:52:31+08:00 | 20260725_122430 | 6D几何量纲消融：log(Cz)改为log(Cz / 1000 mm)，保留当前TableFrame 7D、无crop_cam与V2.2质量门控。 | 21.622 mm | +0.839 mm | 263 |
| 2026-07-28T10:59:01+08:00 | 20260728_103423 | 6D几何消融：第三维使用固定范围线性深度(Cz-500)/(1000-500)，保留当前TableFrame 7D、无crop_cam与V2.2质量门控。 | 21.228 mm | -0.394 mm | 263 |
| 2026-07-28T19:56:09+08:00 | 20260728_193536 | 6D几何消融：第三维使用固定范围线性深度(Cz-600)/(850-600)，保留当前TableFrame 7D、无crop_cam与V2.2质量门控。 | 21.550 mm | +0.322 mm | 263 |
| 2026-07-28T20:53:28+08:00 | 20260728_203116 | 6D几何消融：第三维使用线性绝对深度Cz/100 mm，保留当前TableFrame 7D、无crop_cam与V2.2质量门控。 | 21.266 mm | -0.284 mm | 287 |
| 2026-07-28T21:27:39+08:00 | 20260728_210429 | 当前配置：TableFrame 7D、无crop_cam与V2.2质量门控；双眼6D第三维使用log(Cz)。 | 20.783 mm | -0.483 mm | 248 |
| 2026-07-29T00:12:49+08:00 | 20260728_232915 | V3：在V2.2基础上增加RGB人脸ResNet18编码器；face与双眼特征先融合为128D视觉特征，再与DECA、TableFrame和质量门控双眼几何拼接。 | 23.271 mm | +2.487 mm | 232 |
| 2026-07-29T10:33:11+08:00 | 20260729_100651 | 当前配置：TableFrame 7D、无crop_cam与V2.2质量门控；双眼6D第三维使用log(Cz)。 | 20.783 mm | -2.487 mm | 248 |
| 2026-07-29T11:05:39+08:00 | 20260729_104153 | 当前配置：TableFrame 7D、无crop_cam与V2.2质量门控；双眼6D第三维使用log(Cz)。 | 20.783 mm | +0.000 mm | 248 |
| 2026-07-29T14:49:23+08:00 | 20260729_142525 | DECA消融：仅使用shape+exp+pose的Geometry 156D特征；保留TableFrame 7D、无crop_cam与V2.2质量门控。 | 21.363 mm | +0.580 mm | 210 |
| 2026-07-29T16:36:13+08:00 | 20260729_161308 | 几何表示消融：双眼先验从中心—基线6D切换为原始左右眼相机坐标RawEye 6D；保留DECA Geometry156、TableFrame7D、无crop_cam与V2.2质量门控。 | 20.796 mm | -0.567 mm | 246 |
| 2026-07-29T19:58:22+08:00 | 20260729_192939 | DECA分解编码消融：shape、exp、pose分别经小型MLP后拼接融合为128D；保留RawEye 6D、TableFrame7D、无crop_cam与V2.2质量门控。 | 22.311 mm | +1.515 mm | 248 |
| 2026-07-29T20:35:55+08:00 | 20260729_200634 | DECA分解编码消融：shape、exp、pose分别经小型MLP后拼接融合为128D；保留RawEye 6D、TableFrame7D、无crop_cam与V2.2质量门控。 | 21.944 mm | -0.367 mm | 248 |
| 2026-07-29T22:17:36+08:00 | 20260729_212830 | V3：在V2.2基础上增加RGB人脸ResNet18编码器；face与双眼特征先融合为128D视觉特征，再与DECA、TableFrame和质量门控双眼几何拼接。 | 23.133 mm | +1.189 mm | 141 |
| 2026-07-30T11:05:12+08:00 | 20260730_102300 | V3：在V2.2基础上增加RGB人脸ResNet18编码器；face与双眼特征先融合为128D视觉特征，再与DECA、TableFrame和质量门控双眼几何拼接。 | 21.093 mm | -2.039 mm | 248 |
| 2026-07-30T15:04:00+08:00 | 20260730_140918 | V3：在V2.2基础上增加RGB人脸ResNet18编码器；face与双眼特征先融合为128D视觉特征，再与DECA、TableFrame和质量门控双眼几何拼接。 | 23.133 mm | +2.039 mm | 141 |
| 2026-07-30T19:41:19+08:00 | 20260730_185157 | V3：在V2.2基础上增加RGB人脸ResNet18编码器；face与双眼特征先融合为128D视觉特征，再与DECA、TableFrame和质量门控双眼几何拼接。 | 23.023 mm | -0.110 mm | 201 |
| 2026-07-30T21:00:48+08:00 | 20260730_202154 | V3：在V2.2基础上增加RGB人脸ResNet18编码器；face与双眼特征先融合为128D视觉特征，再与DECA、TableFrame和质量门控双眼几何拼接。 | 20.796 mm | -2.227 mm | 246 |
| 2026-08-03T08:26:05+08:00 | 20260803_081359 | V4 point-depth ablation: add weak cosine supervision from the PnP-eye-to-Vicon-target pseudo gaze. | 66.797 mm | +46.002 mm | 6 |
| 2026-08-03T10:13:45+08:00 | 20260803_095629 | Controlled camera/session holdout: train only on dataset 3 and validate only on dataset 4; otherwise identical to the RGB-face V4 point+pseudo-gaze baseline. | 78.787 mm | +11.989 mm | 31 |
| 2026-08-03T16:04:09+08:00 | 20260803_154452 | V4 point-depth ablation: add weak cosine supervision from the PnP-eye-to-Vicon-target pseudo gaze. | 45.754 mm | -33.033 mm | 17 |
| 2026-08-03T19:18:27+08:00 | 20260803_185853 | V4 point-depth ablation: add weak cosine supervision from the PnP-eye-to-Vicon-target pseudo gaze. | 45.754 mm | +0.000 mm | 17 |
| 2026-08-03T19:34:32+08:00 | 20260803_192317 | Stage 2 depth correction: load and freeze the dataset3 gaze model, then train only a bounded shared PnP eye scale; validate on dataset4. | 35.684 mm | -10.070 mm | 87 |
| 2026-08-03T21:48:37+08:00 | 20260803_213438 | Controlled camera/session holdout: train only on dataset 3 and validate only on dataset 4; otherwise identical to the RGB-face V4 point+pseudo-gaze baseline. | 64.639 mm | +28.955 mm | 38 |
| 2026-08-03T22:06:05+08:00 | 20260803_215445 | Stage 2 depth correction: load and freeze the dataset3 gaze model, then train only a bounded shared PnP eye scale; validate on dataset4. | 64.649 mm | +0.010 mm | 1 |
| 2026-08-04T10:25:46+08:00 | 20260804_100848 | Controlled camera/session holdout: train only on dataset 3 and validate only on dataset 4; otherwise identical to the RGB-face V4 point+pseudo-gaze baseline. | 63.392 mm | -1.257 mm | 87 |
| 2026-08-04T10:59:04+08:00 | 20260804_104428 | Controlled camera/session holdout: train only on dataset 3 and validate only on dataset 4; otherwise identical to the RGB-face V4 point+pseudo-gaze baseline. | 68.590 mm | +5.198 mm | 52 |
| 2026-08-04T23:19:12+08:00 | dataset5_v4_point_iris_ipd65_common971 | V4 stage 1: image predicts camera-space gaze; one PnP eye midpoint and parameter-free ray/table geometry predict UV. | 41.292 mm | -27.298 mm | 21 |
| 2026-08-05T08:55:02+08:00 | dataset3_4_v4_point_pnp1010_common842 | V4 stage 1: image predicts camera-space gaze; one PnP eye midpoint and parameter-free ray/table geometry predict UV. | 71.784 mm | +30.492 mm | 83 |
| 2026-08-05T09:07:34+08:00 | dataset3_4_v4_point_iris_ipd65_common842 | V4 stage 1: image predicts camera-space gaze; one PnP eye midpoint and parameter-free ray/table geometry predict UV. | 69.063 mm | -2.721 mm | 53 |
| 2026-08-06T12:04:43+08:00 | numbered_train13_val3_v4_iris_ipd_65mm_fixed_prior_20260806_114607 | Auto-generated numbered-artifact split: train=13, val=3, prior=iris_ipd_65mm, depth_mode=fixed_prior. | 83.640 mm | +14.577 mm | 30 |
| 2026-08-07T09:42:52+08:00 | numbered_train3_val4_v4_iris_ipd_65mm_point_20260807_093016 | Auto-generated numbered-artifact split: train=3, val=4, prior=iris_ipd_65mm, depth_mode=point. | 69.342 mm | -14.298 mm | 68 |
| 2026-08-07T11:42:28+08:00 | numbered_train3_val4_v4_iris_ipd_65mm_point_20260807_112933 | Auto-generated numbered-artifact split: train=3, val=4, prior=iris_ipd_65mm, depth_mode=point. | 73.782 mm | +4.440 mm | 56 |
| 2026-08-07T15:31:36+08:00 | numbered_train3_val4_v4_iris_ipd_65mm_point_20260807_151924 | Auto-generated numbered-artifact split: train=3, val=4, prior=iris_ipd_65mm, depth_mode=point. | 69.342 mm | -4.440 mm | 68 |
| 2026-08-07T23:50:26+08:00 | 20260807_233119 | Virtual-camera integration baseline: train dataset 3, validate unseen-camera dataset 4; predict gaze in N and rotate deterministically to C before geometry. DECA is disabled to isolate normalized RGB. | 39.946 mm | -29.395 mm | 58 |
| 2026-08-08T09:51:35+08:00 | numbered_train13_val3_v4_iris_ipd_65mm_point_20260808_092612 | Auto-generated numbered-artifact split: train=13, val=3, prior=iris_ipd_65mm, depth_mode=point. | 83.234 mm | +43.288 mm | 39 |
| 2026-08-08T11:27:27+08:00 | 20260808_105208 | Virtual-camera integration baseline: train dataset 13, validate unseen-camera dataset 3; predict gaze in N and rotate deterministically to C before geometry. DECA is disabled to isolate normalized RGB. | 106.195 mm | +22.961 mm | 31 |
| 2026-08-08T12:16:37+08:00 | 20260808_114943 | Virtual-camera integration baseline: train dataset 11, validate unseen-camera dataset 4; predict gaze in N and rotate deterministically to C before geometry. DECA is disabled to isolate normalized RGB. | 58.620 mm | -47.576 mm | 28 |
| 2026-08-08T21:07:45+08:00 | 20260808_203708 | Virtual-camera integration baseline: train dataset 13, validate unseen-camera dataset 3; predict gaze in N and rotate deterministically to C before geometry. DECA is disabled to isolate normalized RGB. | 106.195 mm | +47.576 mm | 31 |
| 2026-08-08T23:14:10+08:00 | 20260808_225452 | Virtual-camera integration baseline: train dataset 12, validate unseen-camera dataset 4; predict gaze in N and rotate deterministically to C before geometry. DECA is disabled to isolate normalized RGB. | 79.562 mm | -26.633 mm | 34 |
| 2026-08-09T12:25:55+08:00 | 20260809_115900 | Virtual-camera integration baseline: train dataset 4, validate unseen-camera dataset 3; predict gaze in N and rotate deterministically to C before geometry. DECA is disabled to isolate normalized RGB. | 94.361 mm | +14.800 mm | 31 |
| 2026-08-09T17:16:54+08:00 | numbered_train4_val3_v4_iris_ipd_65mm_point_20260809_170619 | Auto-generated numbered-artifact split: train=4, val=3, prior=iris_ipd_65mm, depth_mode=point. | 50.419 mm | -43.942 mm | 35 |
| 2026-08-09T19:23:51+08:00 | 20260809_185006 | Strict raw-image control for the paired Dataset13-to-Dataset4 virtual-camera experiment: predict gaze directly in camera coordinates while retaining the same sample IDs, Iris/IPD geometry, architecture, loss, and optimization settings. | 84.995 mm | +34.576 mm | 39 |
| 2026-08-09T20:44:13+08:00 | 20260809_200932 | Virtual-camera integration baseline: train dataset 13, validate unseen-camera dataset 4; predict gaze in N and rotate deterministically to C before geometry. DECA is disabled to isolate normalized RGB. | 94.361 mm | +9.366 mm | 31 |
| 2026-08-09T21:25:08+08:00 | 20260809_210949 | Strict raw-image control for the paired Dataset13-to-Dataset4 virtual-camera experiment: predict gaze directly in camera coordinates while retaining the same sample IDs, Iris/IPD geometry, architecture, loss, and optimization settings. | 73.145 mm | -21.216 mm | 20 |
| 2026-08-09T23:04:35+08:00 | 20260809_222018 | Virtual-camera integration baseline: train dataset 13, validate unseen-camera dataset 4; predict gaze in N and rotate deterministically to C before geometry. DECA is disabled to isolate normalized RGB. | 94.361 mm | +21.216 mm | 31 |
| 2026-08-14T19:09:23+08:00 | 20260814_185141 | Preprocessed-v2 virtual-camera experiment: train Dataset 11, validate Dataset 3; select Iris65 or PnP1010 with data.depth_method, predict gaze in N, and rotate deterministically to C before geometry. | 93.587 mm | -0.774 mm | 8 |
| 2026-08-14T20:49:56+08:00 | 20260814_203134 | Preprocessed-v2 virtual-camera experiment: train Dataset 11, validate Dataset 3; select Iris65 or PnP1010 with data.depth_method, predict gaze in N, and rotate deterministically to C before geometry. | 93.393 mm | -0.195 mm | 29 |
| 2026-08-14T22:10:35+08:00 | 20260814_215245 | Preprocessed-v2 strict legacy-image control paired with the virtual-camera experiment: train Dataset 11, validate Dataset 3; use the same Iris65 depth contracts and valid sample IDs, but predict gaze directly in camera coordinates from the original images. | 91.920 mm | -1.472 mm | 14 |
| 2026-08-14T22:50:37+08:00 | 20260814_223316 | Preprocessed-v2 strict legacy-image control paired with the virtual-camera experiment: train Dataset 11, validate Dataset 3; use the same Iris65 depth contracts and valid sample IDs, but predict gaze directly in camera coordinates from the original images. | 91.920 mm | +0.000 mm | 14 |
| 2026-08-14T23:20:20+08:00 | 20260814_230359 | Preprocessed-v2 strict legacy-image control paired with the virtual-camera experiment: train Dataset 11, validate Dataset 3; use the same Iris65 depth contracts and valid sample IDs, but predict gaze directly in camera coordinates from the original images. | 74.039 mm | -17.881 mm | 23 |
| 2026-08-16T00:01:55+08:00 | 20260815_234221 | Preprocessed-v2 direct-UV virtual-camera experiment: train Dataset 11, validate Dataset 3; transform the world gaze target through C into virtual camera N and supervise (x_N, y_N), encode TableFrame7 in N, and condition the learned UV fusion with FiLM from training-normalized log virtual-distance scale. | 62.579 mm | -11.460 mm | 21 |
| 2026-08-16T22:33:19+08:00 | 20260816_220733 | Preprocessed-v2 direct-UV virtual-camera experiment: train Dataset 11, validate Dataset 3; transform the world gaze target through C into virtual camera N and supervise (x_N, y_N), encode TableFrame7 in N, and condition the learned UV fusion with FiLM from training-normalized log virtual-distance scale. | 65.552 mm | +2.973 mm | 26 |
| 2026-08-16T23:21:48+08:00 | 20260816_224737 | Preprocessed-v2 direct-UV virtual-camera experiment: train Dataset 11, validate Dataset 3; transform the world gaze target through C into virtual camera N and supervise (x_N, y_N), encode TableFrame7 in N, and condition the learned UV fusion with FiLM from training-normalized log virtual-distance scale. | 65.448 mm | -0.103 mm | 26 |
| 2026-08-17T00:13:06+08:00 | 20260816_234921 | Preprocessed-v2 direct-UV virtual-camera experiment: train Dataset 11, validate Dataset 3; transform the world gaze target through C into virtual camera N and supervise (x_N, y_N), encode TableFrame7 in N, and condition the learned UV fusion with FiLM from training-normalized log virtual-distance scale. | 65.448 mm | +0.000 mm | 26 |
| 2026-08-17T10:55:00+08:00 | 20260817_101832 | Preprocessed-v2 direct-UV virtual-camera experiment: train Dataset 11, validate Dataset 3; transform the world gaze target through C into virtual camera N and supervise (x_N, y_N), encode TableFrame7 in N, and condition the learned UV fusion with FiLM from training-normalized log virtual-distance scale. | 29.672 mm | -35.776 mm | 20 |
| 2026-08-17T15:22:14+08:00 | 20260817_150211 | Preprocessed-v2 direct-UV virtual-camera experiment: train Dataset 11, validate Dataset 3; transform the world gaze target through C into virtual camera N and supervise (x_N, y_N), encode TableFrame7 in N, and condition the learned UV fusion with FiLM from training-normalized log virtual-distance scale. | 28.117 mm | -1.555 mm | 20 |
| 2026-08-17T15:39:55+08:00 | 20260817_152440 | Preprocessed-v2 direct-UV virtual-camera experiment: train Dataset 11, validate Dataset 3; transform the world gaze target through C into virtual camera N and supervise (x_N, y_N), encode TableFrame7 in N, and condition the learned UV fusion with FiLM from training-normalized log virtual-distance scale. | 27.447 mm | -0.671 mm | 31 |
| 2026-08-17T16:07:15+08:00 | 20260817_155210 | Preprocessed-v2 direct-UV virtual-camera experiment: train Dataset 11, validate Dataset 3; transform the world gaze target through C into virtual camera N and supervise (x_N, y_N), encode TableFrame7 in N, and condition the learned UV fusion with FiLM from training-normalized log virtual-distance scale. | 26.699 mm | -0.748 mm | 36 |
| 2026-08-17T16:36:11+08:00 | 20260817_162644 | Preprocessed-v2 direct-UV virtual-camera experiment: train Dataset 11, validate Dataset 3; transform the world gaze target through C into virtual camera N and supervise (x_N, y_N), encode TableFrame7 in N, and condition the learned UV fusion with FiLM from training-normalized log virtual-distance scale. | 28.988 mm | +2.289 mm | 50 |
| 2026-08-17T18:30:22+08:00 | 20260817_181332 | Preprocessed-v2 direct-UV virtual-camera experiment: train Dataset 11, validate Dataset 3; transform the world gaze target through C into virtual camera N and supervise (x_N, y_N), encode TableFrame7 in N, and condition the learned UV fusion with FiLM from training-normalized log virtual-distance scale. | 30.241 mm | +1.254 mm | 34 |
| 2026-08-17T23:23:26+08:00 | 20260817_230855 | Preprocessed-v2 direct-UV virtual-camera experiment: train Dataset 11, validate Dataset 3; transform the world gaze target through C into virtual camera N and supervise (x_N, y_N), encode TableFrame7 in N, and condition the learned UV fusion with FiLM from training-normalized log virtual-distance scale. | 29.253 mm | -0.988 mm | 29 |
| 2026-08-18T09:30:20+08:00 | 20260818_091627 | Paired virtual-camera gaze-geometry control for the direct-UV experiment: train Dataset 13, validate Dataset 11; use the same Iris65 preprocessed-v2 contract, virtual images, visual backbone, augmentation, batch size, optimizer, schedule, and epoch budget. The only prediction-path difference is gaze_N -> deterministic N-to-C rotation -> ray/table geometry. | 34.991 mm | +5.738 mm | 36 |
| 2026-08-18T10:02:19+08:00 | 20260818_094749 | Preprocessed-v2 direct-UV virtual-camera experiment: output normalized P_N[:2], reconstruct P_N.z on the known per-sample table plane, and compute loss/EPE against table-local UV in millimetres; encode TableFrame7 in N and condition fusion with FiLM from training-normalized log virtual-distance scale. | 41.263 mm | +6.272 mm | 29 |
| 2026-08-18T10:58:09+08:00 | 20260818_104409 | Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from training-normalized log virtual-distance scale. | 36.939 mm | -4.324 mm | 30 |
| 2026-08-18T15:33:01+08:00 | 20260818_151611 | Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from Condition17=[TableFrame7_N, log virtual-distance scale, virtual camera pose in table frame]. | 33.979 mm | -2.960 mm | 25 |
| 2026-08-18T17:05:35+08:00 | 20260818_165029 | Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from Condition17=[TableFrame7_N, log virtual-distance scale, virtual camera pose in table frame]. | 42.016 mm | +8.037 mm | 8 |
| 2026-08-18T23:17:08+08:00 | 20260818_230245 | Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from Condition17=[TableFrame7_N, log virtual-distance scale, virtual camera pose in table frame]. | 39.789 mm | -2.227 mm | 19 |
| 2026-08-19T15:50:46+08:00 | 20260819_153332 | Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from Condition17=[TableFrame7_N, log virtual-distance scale, virtual camera pose in table frame]. | 34.389 mm | -5.399 mm | 25 |
| 2026-08-19T17:12:04+08:00 | 20260819_165805 | Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy eye images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from Condition17=[TableFrame7_N, log virtual-distance scale, virtual camera pose in table frame]. | 32.227 mm | -2.163 mm | 25 |
| 2026-08-20T09:14:55+08:00 | 20260820_085926 | Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy eye images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from Condition17=[TableFrame7_N, log virtual-distance scale, virtual camera pose in table frame]. | 34.238 mm | +2.012 mm | 18 |
| 2026-08-20T10:21:55+08:00 | 20260820_100534 | Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy eye images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from Condition17=[TableFrame7_N, log virtual-distance scale, virtual camera pose in table frame]. | 33.182 mm | -1.056 mm | 45 |
| 2026-08-20T11:03:59+08:00 | 20260820_104613 | Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy eye images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from Condition17=[TableFrame7_N, log virtual-distance scale, virtual camera pose in table frame]. | 33.182 mm | +0.000 mm | 45 |
| 2026-08-20T11:39:34+08:00 | 20260820_112502 | Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy eye images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from Condition17=[TableFrame7_N, log virtual-distance scale, virtual camera pose in table frame]. | 33.182 mm | +0.000 mm | 45 |
| 2026-08-20T15:20:42+08:00 | 20260820_150433 | Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy eye images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from Condition17=[TableFrame7_N, log virtual-distance scale, virtual camera pose in table frame]. | 34.481 mm | +1.299 mm | 19 |
| 2026-08-20T16:03:25+08:00 | 20260820_154852 | Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy eye images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from Condition17=[TableFrame7_N, log virtual-distance scale, virtual camera pose in table frame]. | 34.005 mm | -0.476 mm | 38 |
| 2026-08-20T16:35:09+08:00 | 20260820_161934 | Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy eye images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from Condition17=[TableFrame7_N, log virtual-distance scale, virtual camera pose in table frame]. | 32.787 mm | -1.218 mm | 30 |
| 2026-08-20T19:09:27+08:00 | 20260820_185238 | Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy eye images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from Condition17=[TableFrame7_N, log virtual-distance scale, virtual camera pose in table frame]. | 33.349 mm | +0.562 mm | 41 |
| 2026-08-20T20:25:14+08:00 | 20260820_200821 | Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy eye images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from Condition17=[TableFrame7_N, log virtual-distance scale, virtual camera pose in table frame]. | 33.405 mm | +0.055 mm | 41 |
| 2026-08-20T22:18:58+08:00 | 20260820_220340 | Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy eye images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from Condition17=[TableFrame7_N, log virtual-distance scale, virtual camera pose in table frame]. | 33.284 mm | -0.120 mm | 41 |
| 2026-08-20T23:24:17+08:00 | 20260820_230711 | Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy eye images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from Condition17=[TableFrame7_N, log virtual-distance scale, virtual camera pose in table frame]. | 32.180 mm | -1.104 mm | 42 |
| 2026-08-21T00:08:45+08:00 | 20260820_235025 | Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy eye images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from Condition17=[TableFrame7_N, log virtual-distance scale, virtual camera pose in table frame]. | 33.622 mm | +1.441 mm | 16 |
| 2026-08-21T11:35:21+08:00 | 20260821_111758 | Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy eye images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from Condition17=[TableFrame7_N, log virtual-distance scale, virtual camera pose in table frame]. | 32.151 mm | -1.471 mm | 50 |
| 2026-08-21T15:33:48+08:00 | 20260821_151148 | Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy eye images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from Condition17=[TableFrame7_N, log virtual-distance scale, virtual camera pose in table frame]. | 32.151 mm | +0.000 mm | 50 |
| 2026-08-21T16:23:36+08:00 | 20260821_160335 | Preprocessed-v2 direct table-UV H5+F1 experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy 90x56 eye images; Eye uses H5 small stem + multiscale FPN 4x6 tokens, Face uses F1 padded mixed_7a 4x4 tokens, then direct table-local (u,v) is predicted with virtual-distance and virtual-pose FiLM. | 33.164 mm | +1.013 mm | 45 |
| 2026-08-21T18:25:25+08:00 | 20260821_180729 | Preprocessed-v2 direct table-UV H5+F0 experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy 90x56 eye images; Eye uses H5 small stem + multiscale FPN 4x6 tokens, Face uses the native Inception-ResNet V1 3x3 spatial tokens, then direct table-local (u,v) is predicted with virtual-distance and virtual-pose FiLM. | 32.180 mm | -0.984 mm | 42 |
| 2026-08-23T21:52:25+08:00 | 20260823_212547 | Preprocessed-v2 direct table-UV H5+F0 experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy 90x56 eye images; Eye uses H5 small stem + multiscale FPN 4x6 tokens, Face uses the native Inception-ResNet V1 3x3 spatial tokens, then direct table-local (u,v) is predicted with virtual-distance and virtual-pose FiLM. | 32.085 mm | -0.095 mm | 21 |
| 2026-08-24T10:02:06+08:00 | 20260824_093535 | Preprocessed-v2 direct table-UV H5+F0 experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy 90x56 eye images; Eye uses H5 small stem + multiscale FPN 4x6 tokens, Face uses the native Inception-ResNet V1 3x3 spatial tokens, then direct table-local (u,v) is predicted with virtual-distance and virtual-pose FiLM. | 32.245 mm | +0.161 mm | 37 |
| 2026-08-24T20:59:08+08:00 | 20260824_203558 | Preprocessed-v2 direct table-UV H5+F0 experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy 90x56 eye images; Eye uses H5 small stem + multiscale FPN 4x6 tokens, Face uses the native Inception-ResNet V1 3x3 spatial tokens, then direct table-local (u,v) is predicted with virtual-distance and virtual-pose FiLM. | 32.128 mm | -0.117 mm | 37 |
| 2026-08-25T13:24:05+08:00 | 20260825_130035 | Preprocessed-v2 direct table-UV H5+F0 experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy 90x56 eye images; Eye uses H5 small stem + multiscale FPN 4x6 tokens, Face uses the native Inception-ResNet V1 3x3 spatial tokens, then direct table-local (u,v) is predicted with virtual-distance and virtual-pose FiLM. | 32.578 mm | +0.449 mm | 42 |
| 2026-08-25T14:40:16+08:00 | 20260825_140932 | Preprocessed-v2 direct table-UV H5+F0 experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy 90x56 eye images; Eye uses H5 small stem + multiscale FPN 4x6 tokens, Face uses the native Inception-ResNet V1 3x3 spatial tokens, then direct table-local (u,v) is predicted with virtual-distance and virtual-pose FiLM. | 31.116 mm | -1.462 mm | 43 |
| 2026-08-25T22:39:18+08:00 | 20260825_211030 | Preprocessed-v2 direct table-UV H5+F0 experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy 90x56 eye images; Eye uses H5 small stem + multiscale FPN 4x6 tokens, Face uses the native Inception-ResNet V1 3x3 spatial tokens, then direct table-local (u,v) is predicted with virtual-distance and virtual-pose FiLM. | 32.030 mm | +0.914 mm | 39 |
<!-- AUTO-INDEX-END -->

## 实验详情

## 2026-07-23T17:11:37+08:00 · 20260723_164504

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_100_deca_crop/20260723_164504","completed_at":"2026-07-23T17:11:37+08:00","run_name":"20260723_164504","best_val_epe_mm":20.69390106201172,"best_epoch":217} -->

- 实验目的：V2.2：双眼6D几何加入PnP质量门控与可学习残差修正。
- 模型变化：V2.2：双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_100_deca_crop`，mode=`online`，tags=['ModelV1', 'V2_2', 'eye_geometry_6d', 'quality_gate', 'learned_residual', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_100_deca_crop/20260723_164504`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 眼部骨干 | `resnet18`，weights=`None` |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=11,522,325 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 217） | 0.038 | 20.694 mm | 16.601 mm | 10.365 mm | 15.841 mm |
| 最终训练（epoch 300） | 0.020 | 15.695 mm | 13.774 mm | 10.264 mm | 9.709 mm |
| 最终验证（epoch 300） | 0.039 | 20.984 mm | 17.646 mm | 10.593 mm | 16.026 mm |

相比上一实验：这是自动日记记录的首个实验，暂无上一实验可比较。

## 2026-07-23T18:25:55+08:00 · 20260723_175740

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_100_deca_crop/20260723_175740","completed_at":"2026-07-23T18:25:55+08:00","run_name":"20260723_175740","best_val_epe_mm":21.21085548400879,"best_epoch":269} -->

- 实验目的：V2.2：双眼6D几何加入PnP质量门控与可学习残差修正。
- 模型变化：双眼 6D 几何使用固定 PnP 质量门控。
- W&B：project=`ModelV1_random_80_20_100_deca_crop`，mode=`online`，tags=['ModelV1', 'V2_2', 'eye_geometry_6d', 'quality_gate', 'learned_residual', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_100_deca_crop/20260723_175740`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 眼部骨干 | `resnet18`，weights=`None` |
| 几何分支 | use=True，gate=`fixed` |
| 融合层 | `[256, 128]`；总参数=11,522,228 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 269） | 0.038 | 21.211 mm | 17.680 mm | 10.558 mm | 16.126 mm |
| 最终训练（epoch 300） | 0.022 | 16.462 mm | 14.760 mm | 10.927 mm | 10.132 mm |
| 最终验证（epoch 300） | 0.039 | 21.271 mm | 18.051 mm | 10.705 mm | 16.075 mm |

相比上一实验：最佳 Val EPE 上升 0.517 mm，性能下降。

## 2026-07-23T21:41:08+08:00 · 20260723_211817

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_100_deca_crop/20260723_211817","completed_at":"2026-07-23T21:41:08+08:00","run_name":"20260723_211817","best_val_epe_mm":22.29008674621582,"best_epoch":289} -->

- 实验目的：Crop消融：V2.2关闭crop_cam分支，保留scene、双眼6D几何和质量门控。
- 模型变化：移除 crop_cam 36D 分支；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_100_deca_crop`，mode=`online`，tags=['ModelV1', 'V2_2', 'no_crop_cam', 'eye_geometry_6d', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_100_deca_crop/20260723_211817`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=11,492,493 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 289） | 0.042 | 22.290 mm | 18.103 mm | 11.278 mm | 16.949 mm |
| 最终训练（epoch 300） | 0.022 | 16.413 mm | 14.367 mm | 10.699 mm | 10.325 mm |
| 最终验证（epoch 300） | 0.042 | 22.423 mm | 18.623 mm | 11.384 mm | 17.044 mm |

相比上一实验：最佳 Val EPE 上升 1.079 mm，性能下降。

## 2026-07-23T23:06:05+08:00 · 20260723_224136

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_100_deca_crop/20260723_224136","completed_at":"2026-07-23T23:06:05+08:00","run_name":"20260723_224136","best_val_epe_mm":20.95374298095703,"best_epoch":260} -->

- 实验目的：Scene消融：恢复crop_cam，将Scene从full25替换为物理TableFrame 7D。
- 模型变化：Scene 从 full25 替换为 TableFrame 7D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_100_deca_crop`，mode=`online`，tags=['ModelV1', 'V2_2', 'scene_table_frame7', 'eye_geometry_6d', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_100_deca_crop/20260723_224136`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=True，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=11,519,985 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 260） | 0.038 | 20.954 mm | 17.121 mm | 10.705 mm | 15.743 mm |
| 最终训练（epoch 300） | 0.022 | 16.569 mm | 15.200 mm | 10.925 mm | 10.273 mm |
| 最终验证（epoch 300） | 0.039 | 21.207 mm | 17.562 mm | 10.690 mm | 16.087 mm |

相比上一实验：最佳 Val EPE 下降 1.336 mm，性能提升。

## 2026-07-24T00:09:08+08:00 · 20260723_234608

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_100_deca_crop/20260723_234608","completed_at":"2026-07-24T00:09:08+08:00","run_name":"20260723_234608","best_val_epe_mm":21.947961807250977,"best_epoch":233} -->

- 实验目的：Scene消融：保持crop_cam和V2.2质量门控，将Scene替换为桌面Orientation 6D。
- 模型变化：Scene 替换为桌面 Orientation 6D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_100_deca_crop`，mode=`online`，tags=['ModelV1', 'V2_2', 'scene_orientation6d', 'eye_geometry_6d', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_100_deca_crop/20260723_234608`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=True，input_dim=36 |
| Scene分支 | representation=`orientation6d`，input_dim=6 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=11,519,855 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 233） | 0.040 | 21.948 mm | 17.284 mm | 11.965 mm | 15.753 mm |
| 最终训练（epoch 300） | 0.024 | 17.451 mm | 15.452 mm | 11.399 mm | 10.862 mm |
| 最终验证（epoch 300） | 0.041 | 22.069 mm | 17.949 mm | 11.870 mm | 15.922 mm |

相比上一实验：最佳 Val EPE 上升 0.994 mm，性能下降。

## 2026-07-24T10:22:37+08:00 · 20260724_095707

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_100_deca_crop/20260724_095707","completed_at":"2026-07-24T10:22:37+08:00","run_name":"20260724_095707","best_val_epe_mm":22.113685607910156,"best_epoch":244} -->

- 实验目的：Scene消融：保持crop_cam和V2.2质量门控，将Scene替换为桌面Orientation 6D。
- 模型变化：移除 crop_cam 36D 分支；Scene 替换为桌面 Orientation 6D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_100_deca_crop`，mode=`online`，tags=['ModelV1', 'V2_2', 'scene_orientation6d', 'eye_geometry_6d', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_100_deca_crop/20260724_095707`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`orientation6d`，input_dim=6 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=11,490,023 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 244） | 0.042 | 22.114 mm | 18.253 mm | 10.929 mm | 16.904 mm |
| 最终训练（epoch 300） | 0.023 | 16.349 mm | 14.335 mm | 10.459 mm | 10.344 mm |
| 最终验证（epoch 300） | 0.043 | 22.438 mm | 18.104 mm | 11.198 mm | 17.131 mm |

相比上一实验：最佳 Val EPE 上升 0.166 mm，性能下降。

## 2026-07-24T11:29:41+08:00 · 20260724_110737

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_100_deca_crop/20260724_110737","completed_at":"2026-07-24T11:29:41+08:00","run_name":"20260724_110737","best_val_epe_mm":20.783071517944336,"best_epoch":248} -->

- 实验目的：Scene消融：保持crop_cam和V2.2质量门控，将Scene替换为桌面Orientation 6D。
- 模型变化：移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_100_deca_crop`，mode=`online`，tags=['ModelV1', 'V2_2', 'scene_orientation6d', 'eye_geometry_6d', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_100_deca_crop/20260724_110737`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=11,490,153 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 248） | 0.038 | 20.783 mm | 17.092 mm | 10.427 mm | 15.970 mm |
| 最终训练（epoch 300） | 0.021 | 16.034 mm | 14.394 mm | 10.778 mm | 9.718 mm |
| 最终验证（epoch 300） | 0.039 | 21.105 mm | 16.963 mm | 10.630 mm | 16.047 mm |

相比上一实验：最佳 Val EPE 下降 1.331 mm，性能提升。

## 2026-07-25T12:52:31+08:00 · 20260725_122430

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_100_deca_crop/20260725_122430","completed_at":"2026-07-25T12:52:31+08:00","run_name":"20260725_122430","best_val_epe_mm":21.621980667114258,"best_epoch":263} -->

- 实验目的：6D几何量纲消融：log(Cz)改为log(Cz / 1000 mm)，保留当前TableFrame 7D、无crop_cam与V2.2质量门控。
- 模型变化：移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_100_deca_crop`，mode=`online`，tags=['ModelV1', 'V2_2', 'scene_table_frame7', 'no_crop_cam', 'eye_geometry_6d', 'log_depth_ref_1m', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_100_deca_crop/20260725_122430`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=11,490,153 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 263） | 0.040 | 21.622 mm | 19.004 mm | 11.106 mm | 16.297 mm |
| 最终训练（epoch 300） | 0.021 | 16.091 mm | 14.580 mm | 10.834 mm | 9.734 mm |
| 最终验证（epoch 300） | 0.041 | 21.844 mm | 18.600 mm | 11.072 mm | 16.622 mm |

相比上一实验：最佳 Val EPE 上升 0.839 mm，性能下降。

## 2026-07-28T10:59:01+08:00 · 20260728_103423

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_100_deca_crop/20260728_103423","completed_at":"2026-07-28T10:59:01+08:00","run_name":"20260728_103423","best_val_epe_mm":21.228282928466797,"best_epoch":263} -->

- 实验目的：6D几何消融：第三维使用固定范围线性深度(Cz-500)/(1000-500)，保留当前TableFrame 7D、无crop_cam与V2.2质量门控。
- 模型变化：移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_100_deca_crop`，mode=`online`，tags=['ModelV1', 'V2_2', 'scene_table_frame7', 'no_crop_cam', 'eye_geometry_6d', 'linear_depth_500_1000mm', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_100_deca_crop/20260728_103423`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=11,490,153 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 263） | 0.039 | 21.228 mm | 17.253 mm | 10.884 mm | 15.907 mm |
| 最终训练（epoch 300） | 0.021 | 16.172 mm | 14.193 mm | 11.014 mm | 9.640 mm |
| 最终验证（epoch 300） | 0.040 | 21.451 mm | 17.115 mm | 10.898 mm | 16.167 mm |

相比上一实验：最佳 Val EPE 下降 0.394 mm，性能提升。

## 2026-07-28T19:56:09+08:00 · 20260728_193536

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_100_deca_crop/20260728_193536","completed_at":"2026-07-28T19:56:09+08:00","run_name":"20260728_193536","best_val_epe_mm":21.550310134887695,"best_epoch":263} -->

- 实验目的：6D几何消融：第三维使用固定范围线性深度(Cz-600)/(850-600)，保留当前TableFrame 7D、无crop_cam与V2.2质量门控。
- 模型变化：移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_100_deca_crop`，mode=`online`，tags=['ModelV1', 'V2_2', 'scene_table_frame7', 'no_crop_cam', 'eye_geometry_6d', 'linear_depth_600_850mm', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_100_deca_crop/20260728_193536`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=11,490,153 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 263） | 0.040 | 21.550 mm | 16.618 mm | 10.860 mm | 16.567 mm |
| 最终训练（epoch 300） | 0.021 | 16.199 mm | 14.096 mm | 10.990 mm | 9.726 mm |
| 最终验证（epoch 300） | 0.041 | 21.735 mm | 17.307 mm | 10.861 mm | 16.771 mm |

相比上一实验：最佳 Val EPE 上升 0.322 mm，性能下降。

## 2026-07-28T20:53:28+08:00 · 20260728_203116

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_100_deca_crop/20260728_203116","completed_at":"2026-07-28T20:53:28+08:00","run_name":"20260728_203116","best_val_epe_mm":21.26618766784668,"best_epoch":287} -->

- 实验目的：6D几何消融：第三维使用线性绝对深度Cz/100 mm，保留当前TableFrame 7D、无crop_cam与V2.2质量门控。
- 模型变化：移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_100_deca_crop`，mode=`online`，tags=['ModelV1', 'V2_2', 'scene_table_frame7', 'no_crop_cam', 'eye_geometry_6d', 'linear_depth_div100mm', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_100_deca_crop/20260728_203116`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=11,490,153 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 287） | 0.039 | 21.266 mm | 17.454 mm | 10.697 mm | 16.167 mm |
| 最终训练（epoch 300） | 0.021 | 16.104 mm | 14.308 mm | 10.816 mm | 9.659 mm |
| 最终验证（epoch 300） | 0.040 | 21.466 mm | 17.358 mm | 10.629 mm | 16.365 mm |

相比上一实验：最佳 Val EPE 下降 0.284 mm，性能提升。

## 2026-07-28T21:27:39+08:00 · 20260728_210429

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_100_deca_crop/20260728_210429","completed_at":"2026-07-28T21:27:39+08:00","run_name":"20260728_210429","best_val_epe_mm":20.783071517944336,"best_epoch":248} -->

- 实验目的：当前配置：TableFrame 7D、无crop_cam与V2.2质量门控；双眼6D第三维使用log(Cz)。
- 模型变化：移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_100_deca_crop`，mode=`online`，tags=['ModelV1', 'V2_2', 'scene_table_frame7', 'no_crop_cam', 'eye_geometry_6d', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_100_deca_crop/20260728_210429`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=11,490,153 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 248） | 0.038 | 20.783 mm | 17.092 mm | 10.427 mm | 15.970 mm |
| 最终训练（epoch 300） | 0.021 | 16.034 mm | 14.394 mm | 10.778 mm | 9.718 mm |
| 最终验证（epoch 300） | 0.039 | 21.105 mm | 16.963 mm | 10.630 mm | 16.047 mm |

相比上一实验：最佳 Val EPE 下降 0.483 mm，性能提升。

## 2026-07-29T00:12:49+08:00 · 20260728_232915

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_100_deca_crop/20260728_232915","completed_at":"2026-07-29T00:12:49+08:00","run_name":"20260728_232915","best_val_epe_mm":23.270545959472656,"best_epoch":232} -->

- 实验目的：V3：在V2.2基础上增加RGB人脸ResNet18编码器；face与双眼特征先融合为128D视觉特征，再与DECA、TableFrame和质量门控双眼几何拼接。
- 模型变化：增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_100_deca_crop`，mode=`online`，tags=['ModelV1', 'V3', 'face_image_resnet18', 'face_eye_visual_fusion', 'scene_table_frame7', 'no_crop_cam', 'eye_geometry_6d', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_100_deca_crop/20260728_232915`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 人脸图像骨干 | use=True，`resnet18`，weights=`DEFAULT`，freeze_until=`layer3` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=22,832,553 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 232） | 0.046 | 23.271 mm | 19.386 mm | 10.313 mm | 18.740 mm |
| 最终训练（epoch 300） | 0.020 | 15.512 mm | 13.660 mm | 10.072 mm | 9.651 mm |
| 最终验证（epoch 300） | 0.047 | 23.732 mm | 19.746 mm | 10.189 mm | 19.365 mm |

相比上一实验：最佳 Val EPE 上升 2.487 mm，性能下降。

## 2026-07-29T10:33:11+08:00 · 20260729_100651

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_100_deca_crop/20260729_100651","completed_at":"2026-07-29T10:33:11+08:00","run_name":"20260729_100651","best_val_epe_mm":20.783071517944336,"best_epoch":248} -->

- 实验目的：当前配置：TableFrame 7D、无crop_cam与V2.2质量门控；双眼6D第三维使用log(Cz)。
- 模型变化：移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_100_deca_crop`，mode=`online`，tags=['ModelV1', 'V2_2', 'scene_table_frame7', 'no_crop_cam', 'eye_geometry_6d', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_100_deca_crop/20260729_100651`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 人脸图像骨干 | use=False，`resnet18`，weights=`None`，freeze_until=`none` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=11,490,153 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 248） | 0.038 | 20.783 mm | 17.092 mm | 10.427 mm | 15.970 mm |
| 最终训练（epoch 300） | 0.021 | 16.034 mm | 14.394 mm | 10.778 mm | 9.718 mm |
| 最终验证（epoch 300） | 0.039 | 21.105 mm | 16.963 mm | 10.630 mm | 16.047 mm |

相比上一实验：最佳 Val EPE 下降 2.487 mm，性能提升。

## 2026-07-29T11:05:39+08:00 · 20260729_104153

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_100_V3/20260729_104153","completed_at":"2026-07-29T11:05:39+08:00","run_name":"20260729_104153","best_val_epe_mm":20.783071517944336,"best_epoch":248} -->

- 实验目的：当前配置：TableFrame 7D、无crop_cam与V2.2质量门控；双眼6D第三维使用log(Cz)。
- 模型变化：移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_100_V3`，mode=`online`，tags=['ModelV1', 'V2_2', 'scene_table_frame7', 'no_crop_cam', 'eye_geometry_6d', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_100_V3/20260729_104153`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 人脸图像骨干 | use=False，`resnet18`，weights=`None`，freeze_until=`none` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=11,490,153 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 248） | 0.038 | 20.783 mm | 17.092 mm | 10.427 mm | 15.970 mm |
| 最终训练（epoch 300） | 0.021 | 16.034 mm | 14.394 mm | 10.778 mm | 9.718 mm |
| 最终验证（epoch 300） | 0.039 | 21.105 mm | 16.963 mm | 10.630 mm | 16.047 mm |

相比上一实验：最佳 Val EPE 与上一实验相同。

## 2026-07-29T14:49:23+08:00 · 20260729_142525

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_100_deca_crop/20260729_142525","completed_at":"2026-07-29T14:49:23+08:00","run_name":"20260729_142525","best_val_epe_mm":21.36295509338379,"best_epoch":210} -->

- 实验目的：DECA消融：仅使用shape+exp+pose的Geometry 156D特征；保留TableFrame 7D、无crop_cam与V2.2质量门控。
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_100_deca_crop`，mode=`online`，tags=['ModelV1', 'V2_2', 'deca_geometry156', 'scene_table_frame7', 'no_crop_cam', 'eye_geometry_6d', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_100_deca_crop/20260729_142525`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 人脸图像骨干 | use=False，`resnet18`，weights=`None`，freeze_until=`none` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=11,469,513 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 210） | 0.040 | 21.363 mm | 17.607 mm | 11.189 mm | 15.994 mm |
| 最终训练（epoch 300） | 0.021 | 16.261 mm | 14.142 mm | 10.616 mm | 10.083 mm |
| 最终验证（epoch 300） | 0.041 | 21.863 mm | 18.474 mm | 10.986 mm | 16.639 mm |

相比上一实验：最佳 Val EPE 上升 0.580 mm，性能下降。

## 2026-07-29T16:36:13+08:00 · 20260729_161308

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_100_deca_crop/20260729_161308","completed_at":"2026-07-29T16:36:13+08:00","run_name":"20260729_161308","best_val_epe_mm":20.79582977294922,"best_epoch":246} -->

- 实验目的：几何表示消融：双眼先验从中心—基线6D切换为原始左右眼相机坐标RawEye 6D；保留DECA Geometry156、TableFrame7D、无crop_cam与V2.2质量门控。
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；双眼几何从中心—基线 6D 切换为原始左右眼相机坐标 RawEye 6D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_100_deca_crop`，mode=`online`，tags=['ModelV1', 'V2_2', 'deca_geometry156', 'scene_table_frame7', 'no_crop_cam', 'eye_geometry_raw_eye6d', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_100_deca_crop/20260729_161308`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 人脸图像骨干 | use=False，`resnet18`，weights=`None`，freeze_until=`none` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=11,469,513 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 246） | 0.038 | 20.796 mm | 16.225 mm | 10.520 mm | 15.802 mm |
| 最终训练（epoch 300） | 0.021 | 16.119 mm | 14.408 mm | 10.566 mm | 10.012 mm |
| 最终验证（epoch 300） | 0.039 | 20.997 mm | 16.370 mm | 10.362 mm | 16.232 mm |

相比上一实验：最佳 Val EPE 下降 0.567 mm，性能提升。

## 2026-07-29T19:58:22+08:00 · 20260729_192939

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_100_deca_crop/20260729_192939","completed_at":"2026-07-29T19:58:22+08:00","run_name":"20260729_192939","best_val_epe_mm":22.311243057250977,"best_epoch":248} -->

- 实验目的：DECA分解编码消融：shape、exp、pose分别经小型MLP后拼接融合为128D；保留RawEye 6D、TableFrame7D、无crop_cam与V2.2质量门控。
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；DECA 的 shape、exp、pose 分别经小型 MLP 编码后再融合为 128D；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；双眼几何从中心—基线 6D 切换为原始左右眼相机坐标 RawEye 6D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_100_deca_crop`，mode=`online`，tags=['ModelV1', 'V2_2', 'deca_geometry156', 'deca_factorized_geometry', 'scene_table_frame7', 'no_crop_cam', 'eye_geometry_raw_eye6d', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_100_deca_crop/20260729_192939`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 人脸图像骨干 | use=False，`resnet18`，weights=`None`，freeze_until=`none` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=11,502,537 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 248） | 0.042 | 22.311 mm | 18.966 mm | 10.914 mm | 16.973 mm |
| 最终训练（epoch 300） | 0.022 | 16.699 mm | 14.828 mm | 11.019 mm | 10.386 mm |
| 最终验证（epoch 300） | 0.043 | 22.578 mm | 19.107 mm | 10.960 mm | 17.081 mm |

相比上一实验：最佳 Val EPE 上升 1.515 mm，性能下降。

## 2026-07-29T20:35:55+08:00 · 20260729_200634

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_100_deca_crop/20260729_200634","completed_at":"2026-07-29T20:35:55+08:00","run_name":"20260729_200634","best_val_epe_mm":21.943836212158203,"best_epoch":248} -->

- 实验目的：DECA分解编码消融：shape、exp、pose分别经小型MLP后拼接融合为128D；保留RawEye 6D、TableFrame7D、无crop_cam与V2.2质量门控。
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；DECA 的 shape、exp、pose 分别经小型 MLP 编码后再融合为 128D；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_100_deca_crop`，mode=`online`，tags=['ModelV1', 'V2_2', 'deca_geometry156', 'deca_factorized_geometry', 'scene_table_frame7', 'no_crop_cam', 'eye_geometry_raw_eye6d', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_100_deca_crop/20260729_200634`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 人脸图像骨干 | use=False，`resnet18`，weights=`None`，freeze_until=`none` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=11,502,537 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 248） | 0.041 | 21.944 mm | 18.444 mm | 10.999 mm | 16.722 mm |
| 最终训练（epoch 300） | 0.023 | 16.874 mm | 15.305 mm | 11.219 mm | 10.464 mm |
| 最终验证（epoch 300） | 0.042 | 22.573 mm | 19.228 mm | 11.277 mm | 17.144 mm |

相比上一实验：最佳 Val EPE 下降 0.367 mm，性能提升。

## 2026-07-29T22:17:36+08:00 · 20260729_212830

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_V3/20260729_212830","completed_at":"2026-07-29T22:17:36+08:00","run_name":"20260729_212830","best_val_epe_mm":23.132816314697266,"best_epoch":141} -->

- 实验目的：V3：在V2.2基础上增加RGB人脸ResNet18编码器；face与双眼特征先融合为128D视觉特征，再与DECA、TableFrame和质量门控双眼几何拼接。
- 模型变化：增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；双眼几何从中心—基线 6D 切换为原始左右眼相机坐标 RawEye 6D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_V3`，mode=`online`，tags=['ModelV1', 'V3', 'face_image_resnet18', 'face_eye_visual_fusion', 'scene_table_frame7', 'no_crop_cam', 'eye_geometry_6d', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_V3/20260729_212830`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 人脸图像骨干 | use=True，`resnet18`，weights=`DEFAULT`，freeze_until=`layer3` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=22,832,553 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 141） | 0.045 | 23.133 mm | 18.122 mm | 10.380 mm | 18.323 mm |
| 最终训练（epoch 300） | 0.020 | 15.613 mm | 14.078 mm | 10.179 mm | 9.867 mm |
| 最终验证（epoch 300） | 0.047 | 23.746 mm | 19.983 mm | 10.083 mm | 19.413 mm |

相比上一实验：最佳 Val EPE 上升 1.189 mm，性能下降。

## 2026-07-30T11:05:12+08:00 · 20260730_102300

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_V3/20260730_102300","completed_at":"2026-07-30T11:05:12+08:00","run_name":"20260730_102300","best_val_epe_mm":21.093454360961914,"best_epoch":248} -->

- 实验目的：V3：在V2.2基础上增加RGB人脸ResNet18编码器；face与双眼特征先融合为128D视觉特征，再与DECA、TableFrame和质量门控双眼几何拼接。
- 模型变化：移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；双眼几何从中心—基线 6D 切换为原始左右眼相机坐标 RawEye 6D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_V3`，mode=`online`，tags=['ModelV1', 'V3', 'face_image_resnet18', 'face_eye_visual_fusion', 'scene_table_frame7', 'no_crop_cam', 'eye_geometry_6d', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_V3/20260730_102300`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 人脸图像骨干 | use=False，`resnet18`，weights=`DEFAULT`，freeze_until=`layer3` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=11,490,153 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 248） | 0.039 | 21.093 mm | 17.326 mm | 10.610 mm | 15.941 mm |
| 最终训练（epoch 300） | 0.021 | 16.074 mm | 14.339 mm | 10.982 mm | 9.439 mm |
| 最终验证（epoch 300） | 0.041 | 21.549 mm | 17.619 mm | 10.760 mm | 16.347 mm |

相比上一实验：最佳 Val EPE 下降 2.039 mm，性能提升。

## 2026-07-30T15:04:00+08:00 · 20260730_140918

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_V3/20260730_140918","completed_at":"2026-07-30T15:04:00+08:00","run_name":"20260730_140918","best_val_epe_mm":23.132816314697266,"best_epoch":141} -->

- 实验目的：V3：在V2.2基础上增加RGB人脸ResNet18编码器；face与双眼特征先融合为128D视觉特征，再与DECA、TableFrame和质量门控双眼几何拼接。
- 模型变化：增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；双眼几何从中心—基线 6D 切换为原始左右眼相机坐标 RawEye 6D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_V3`，mode=`online`，tags=['ModelV1', 'V3', 'deca_geometry156', 'deca_factorized_geometry', 'face_image_resnet18', 'face_eye_visual_fusion', 'scene_table_frame7', 'no_crop_cam', 'eye_geometry_raw_eye6d', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_V3/20260730_140918`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 人脸图像骨干 | use=True，`resnet18`，weights=`DEFAULT`，freeze_until=`layer3` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=22,832,553 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 141） | 0.045 | 23.133 mm | 18.122 mm | 10.380 mm | 18.323 mm |
| 最终训练（epoch 300） | 0.020 | 15.613 mm | 14.078 mm | 10.179 mm | 9.867 mm |
| 最终验证（epoch 300） | 0.047 | 23.746 mm | 19.983 mm | 10.083 mm | 19.413 mm |

相比上一实验：最佳 Val EPE 上升 2.039 mm，性能下降。

## 2026-07-30T19:41:19+08:00 · 20260730_185157

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_V3/20260730_185157","completed_at":"2026-07-30T19:41:19+08:00","run_name":"20260730_185157","best_val_epe_mm":23.022624969482422,"best_epoch":201} -->

- 实验目的：V3：在V2.2基础上增加RGB人脸ResNet18编码器；face与双眼特征先融合为128D视觉特征，再与DECA、TableFrame和质量门控双眼几何拼接。
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；双眼几何从中心—基线 6D 切换为原始左右眼相机坐标 RawEye 6D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_V3`，mode=`online`，tags=['ModelV1', 'V3', 'deca_geometry156', 'deca_factorized_geometry', 'face_image_resnet18', 'face_eye_visual_fusion', 'scene_table_frame7', 'no_crop_cam', 'eye_geometry_raw_eye6d', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_V3/20260730_185157`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 人脸图像骨干 | use=True，`resnet18`，weights=`DEFAULT`，freeze_until=`layer3` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=22,811,913 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 201） | 0.045 | 23.023 mm | 19.736 mm | 11.107 mm | 18.152 mm |
| 最终训练（epoch 300） | 0.021 | 16.332 mm | 15.073 mm | 10.615 mm | 10.114 mm |
| 最终验证（epoch 300） | 0.046 | 23.475 mm | 20.185 mm | 11.010 mm | 18.571 mm |

相比上一实验：最佳 Val EPE 下降 0.110 mm，性能提升。

## 2026-07-30T21:00:48+08:00 · 20260730_202154

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_V3/20260730_202154","completed_at":"2026-07-30T21:00:48+08:00","run_name":"20260730_202154","best_val_epe_mm":20.79582977294922,"best_epoch":246} -->

- 实验目的：V3：在V2.2基础上增加RGB人脸ResNet18编码器；face与双眼特征先融合为128D视觉特征，再与DECA、TableFrame和质量门控双眼几何拼接。
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；双眼几何从中心—基线 6D 切换为原始左右眼相机坐标 RawEye 6D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_V3`，mode=`online`，tags=['ModelV1', 'V3', 'deca_geometry156', 'deca_factorized_geometry', 'face_image_resnet18', 'face_eye_visual_fusion', 'scene_table_frame7', 'no_crop_cam', 'eye_geometry_raw_eye6d', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_V3/20260730_202154`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 人脸图像骨干 | use=False，`resnet18`，weights=`DEFAULT`，freeze_until=`layer3` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=11,469,513 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 246） | 0.038 | 20.796 mm | 16.225 mm | 10.520 mm | 15.802 mm |
| 最终训练（epoch 300） | 0.021 | 16.119 mm | 14.408 mm | 10.566 mm | 10.012 mm |
| 最终验证（epoch 300） | 0.039 | 20.997 mm | 16.370 mm | 10.362 mm | 16.232 mm |

相比上一实验：最佳 Val EPE 下降 2.227 mm，性能提升。

## 2026-08-03T08:26:05+08:00 · 20260803_081359

<!-- experiment-record: {"run_dir":"outputs/ModelV1_dataset5_V4/20260803_081359","completed_at":"2026-08-03T08:26:05+08:00","run_name":"20260803_081359","best_val_epe_mm":66.7973403930664,"best_epoch":6} -->

- 实验目的：V4 point-depth ablation: add weak cosine supervision from the PnP-eye-to-Vicon-target pseudo gaze.
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_dataset5_V4`，mode=`online`，tags=['ModelV1', 'V4', 'dataset_5', 'gaze_geometry', 'point_pnp', 'pseudo_gaze_cosine_100', 'imagenet_resnet18', 'eye_augmentation']
- 产物目录：`outputs/ModelV1_dataset5_V4/20260803_081359`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `dataset_5`；train=904，val=229，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 人脸图像骨干 | use=False，`resnet18`，weights=`None`，freeze_until=`none` |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=11,441,691 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=0.0，gaze_angular_weight=100.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 6） | 30.885 | 66.797 mm | 58.004 mm | 40.834 mm | 45.708 mm |
| 最终训练（epoch 100） | 8.265 | 28.573 mm | 25.732 mm | 18.083 mm | 18.212 mm |
| 最终验证（epoch 100） | 41.946 | 82.022 mm | 77.584 mm | 55.292 mm | 52.658 mm |

相比上一实验：最佳 Val EPE 上升 46.002 mm，性能下降。

## 2026-08-03T10:13:45+08:00 · 20260803_095629

<!-- experiment-record: {"run_dir":"outputs/ModelV1_dataset3_train_dataset4_val_V4/20260803_095629","completed_at":"2026-08-03T10:13:45+08:00","run_name":"20260803_095629","best_val_epe_mm":78.78666687011719,"best_epoch":31} -->

- 实验目的：Controlled camera/session holdout: train only on dataset 3 and validate only on dataset 4; otherwise identical to the RGB-face V4 point+pseudo-gaze baseline.
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_dataset3_train_dataset4_val_V4`，mode=`online`，tags=['ModelV1', 'V4', 'train_dataset_3', 'val_dataset_4', 'explicit_holdout', 'gaze_geometry', 'point_pnp', 'pseudo_gaze_cosine_100', 'rgb_face', 'imagenet_resnet18', 'eye_augmentation']
- 产物目录：`outputs/ModelV1_dataset3_train_dataset4_val_V4/20260803_095629`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=506，val=398，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 人脸图像骨干 | use=True，`resnet18`，weights=`None`，freeze_until=`none` |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=22,784,091 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=0.0，gaze_angular_weight=100.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 31） | 37.981 | 78.787 mm | 73.948 mm | 47.770 mm | 53.085 mm |
| 最终训练（epoch 100） | 7.288 | 26.202 mm | 23.722 mm | 16.042 mm | 17.192 mm |
| 最终验证（epoch 100） | 43.957 | 87.920 mm | 83.778 mm | 56.349 mm | 56.380 mm |

相比上一实验：最佳 Val EPE 上升 11.989 mm，性能下降。

## 2026-08-03T16:04:09+08:00 · 20260803_154452

<!-- experiment-record: {"run_dir":"outputs/ModelV1_dataset5_V4/20260803_154452","completed_at":"2026-08-03T16:04:09+08:00","run_name":"20260803_154452","best_val_epe_mm":45.75394821166992,"best_epoch":17} -->

- 实验目的：V4 point-depth ablation: add weak cosine supervision from the PnP-eye-to-Vicon-target pseudo gaze.
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_dataset5_V4`，mode=`online`，tags=['ModelV1', 'V4', 'dataset_5', 'gaze_geometry', 'point_pnp', 'pseudo_gaze_cosine_100', 'imagenet_resnet18', 'eye_augmentation']
- 产物目录：`outputs/ModelV1_dataset5_V4/20260803_154452`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `dataset_5`；train=904，val=229，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v2.csv` |
| 人脸图像骨干 | use=True，`resnet18`，weights=`DEFAULT`，freeze_until=`none` |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=22,784,091 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.1，gaze_direction_weight=0.0，gaze_angular_weight=100.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 17） | 18.613 | 45.754 mm | 39.142 mm | 25.947 mm | 31.846 mm |
| 最终训练（epoch 100） | 5.790 | 21.340 mm | 19.902 mm | 12.463 mm | 14.682 mm |
| 最终验证（epoch 100） | 24.481 | 56.151 mm | 54.834 mm | 39.040 mm | 32.651 mm |

相比上一实验：最佳 Val EPE 下降 33.033 mm，性能提升。

## 2026-08-03T19:18:27+08:00 · 20260803_185853

<!-- experiment-record: {"run_dir":"outputs/ModelV1_dataset5_V4/20260803_185853","completed_at":"2026-08-03T19:18:27+08:00","run_name":"20260803_185853","best_val_epe_mm":45.75394821166992,"best_epoch":17} -->

- 实验目的：V4 point-depth ablation: add weak cosine supervision from the PnP-eye-to-Vicon-target pseudo gaze.
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_dataset5_V4`，mode=`online`，tags=['ModelV1', 'V4', 'dataset_5', 'gaze_geometry', 'point_pnp', 'pseudo_gaze_cosine_100', 'imagenet_resnet18', 'eye_augmentation']
- 产物目录：`outputs/ModelV1_dataset5_V4/20260803_185853`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `dataset_5`；train=904，val=229，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v2.csv` |
| 人脸图像骨干 | use=True，`resnet18`，weights=`DEFAULT`，freeze_until=`none` |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=22,784,091 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.1，gaze_direction_weight=0.0，gaze_angular_weight=100.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 17） | 18.613 | 45.754 mm | 39.142 mm | 25.947 mm | 31.846 mm |
| 最终训练（epoch 100） | 5.790 | 21.340 mm | 19.902 mm | 12.463 mm | 14.682 mm |
| 最终验证（epoch 100） | 24.481 | 56.151 mm | 54.834 mm | 39.040 mm | 32.651 mm |

相比上一实验：最佳 Val EPE 与上一实验相同。

## 2026-08-03T19:34:32+08:00 · 20260803_192317

<!-- experiment-record: {"run_dir":"outputs/ModelV1_dataset3_train_dataset4_val_V4_depth_correction/20260803_192317","completed_at":"2026-08-03T19:34:32+08:00","run_name":"20260803_192317","best_val_epe_mm":35.684207916259766,"best_epoch":87} -->

- 实验目的：Stage 2 depth correction: load and freeze the dataset3 gaze model, then train only a bounded shared PnP eye scale; validate on dataset4.
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_dataset3_train_dataset4_val_V4_depth_correction`，mode=`online`，tags=['ModelV1', 'V4', 'train_dataset_3', 'val_dataset_4', 'depth_correction', 'shared_scale', 'PnP_10D', 'K1', 'frozen_gaze']
- 产物目录：`outputs/ModelV1_dataset3_train_dataset4_val_V4_depth_correction/20260803_192317`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=506，val=398，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v2.csv` |
| 人脸图像骨干 | use=True，`resnet18`，weights=`DEFAULT`，freeze_until=`none` |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=22,786,908 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0001，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=0.0，gaze_angular_weight=0.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 87） | 11.314 | 35.684 mm | 34.263 mm | 27.551 mm | 17.425 mm |
| 最终训练（epoch 100） | 10.543 | 33.557 mm | 31.046 mm | 23.319 mm | 19.414 mm |
| 最终验证（epoch 100） | 11.315 | 35.685 mm | 34.263 mm | 27.552 mm | 17.426 mm |

相比上一实验：最佳 Val EPE 下降 10.070 mm，性能提升。

## 2026-08-03T21:48:37+08:00 · 20260803_213438

<!-- experiment-record: {"run_dir":"outputs/ModelV1_dataset3_train_dataset4_val_V4/20260803_213438","completed_at":"2026-08-03T21:48:37+08:00","run_name":"20260803_213438","best_val_epe_mm":64.63905334472656,"best_epoch":38} -->

- 实验目的：Controlled camera/session holdout: train only on dataset 3 and validate only on dataset 4; otherwise identical to the RGB-face V4 point+pseudo-gaze baseline.
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_dataset3_train_dataset4_val_V4`，mode=`online`，tags=['ModelV1', 'V4', 'train_dataset_3', 'val_dataset_4', 'explicit_holdout', 'gaze_geometry', 'point_pnp', 'pseudo_gaze_cosine_100', 'rgb_face', 'imagenet_resnet18', 'eye_augmentation']
- 产物目录：`outputs/ModelV1_dataset3_train_dataset4_val_V4/20260803_213438`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=506，val=398，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 人脸图像骨干 | use=True，`resnet18`，weights=`DEFAULT`，freeze_until=`none` |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=22,784,091 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=0.0，gaze_angular_weight=100.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 38） | 27.809 | 64.639 mm | 59.384 mm | 22.436 mm | 57.429 mm |
| 最终训练（epoch 100） | 6.606 | 24.974 mm | 22.215 mm | 14.984 mm | 16.608 mm |
| 最终验证（epoch 100） | 28.906 | 66.934 mm | 63.512 mm | 20.999 mm | 60.918 mm |

相比上一实验：最佳 Val EPE 上升 28.955 mm，性能下降。

## 2026-08-03T22:06:05+08:00 · 20260803_215445

<!-- experiment-record: {"run_dir":"outputs/ModelV1_dataset3_train_dataset4_val_V4_depth_correction/20260803_215445","completed_at":"2026-08-03T22:06:05+08:00","run_name":"20260803_215445","best_val_epe_mm":64.6488265991211,"best_epoch":1} -->

- 实验目的：Stage 2 depth correction: load and freeze the dataset3 gaze model, then train only a bounded shared PnP eye scale; validate on dataset4.
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_dataset3_train_dataset4_val_V4_depth_correction`，mode=`online`，tags=['ModelV1', 'V4', 'train_dataset_3', 'val_dataset_4', 'depth_correction', 'shared_scale', 'PnP_10D', 'K1', 'frozen_gaze']
- 产物目录：`outputs/ModelV1_dataset3_train_dataset4_val_V4_depth_correction/20260803_215445`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=506，val=398，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v2.csv` |
| 人脸图像骨干 | use=True，`resnet18`，weights=`DEFAULT`，freeze_until=`none` |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=22,786,908 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0001，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=0.0，gaze_angular_weight=0.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 1） | 27.388 | 64.649 mm | 59.384 mm | 22.433 mm | 57.437 mm |
| 最终训练（epoch 100） | 2.621 | 15.542 mm | 13.838 mm | 10.420 mm | 9.396 mm |
| 最终验证（epoch 100） | 29.933 | 69.496 mm | 65.063 mm | 22.141 mm | 62.777 mm |

相比上一实验：最佳 Val EPE 上升 0.010 mm，性能下降。

## 2026-08-04T10:25:46+08:00 · 20260804_100848

<!-- experiment-record: {"run_dir":"outputs/ModelV1_dataset3_train_dataset4_val_V4/20260804_100848","completed_at":"2026-08-04T10:25:46+08:00","run_name":"20260804_100848","best_val_epe_mm":63.39213562011719,"best_epoch":87} -->

- 实验目的：Controlled camera/session holdout: train only on dataset 3 and validate only on dataset 4; otherwise identical to the RGB-face V4 point+pseudo-gaze baseline.
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_dataset3_train_dataset4_val_V4`，mode=`online`，tags=['ModelV1', 'V4', 'train_dataset_3', 'val_dataset_4', 'explicit_holdout', 'gaze_geometry', 'point_pnp', 'pseudo_gaze_cosine_100', 'rgb_face', 'imagenet_resnet18', 'eye_augmentation']
- 产物目录：`outputs/ModelV1_dataset3_train_dataset4_val_V4/20260804_100848`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=506，val=398，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 人脸图像骨干 | use=True，`resnet18`，weights=`DEFAULT`，freeze_until=`none` |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=22,784,478 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=0.0，gaze_angular_weight=100.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 87） | 39.316 | 63.392 mm | 58.346 mm | 20.508 mm | 57.075 mm |
| 最终训练（epoch 100） | 14.852 | 23.742 mm | 21.218 mm | 14.952 mm | 15.286 mm |
| 最终验证（epoch 100） | 39.721 | 63.902 mm | 59.279 mm | 20.735 mm | 57.550 mm |

相比上一实验：最佳 Val EPE 下降 1.257 mm，性能提升。

## 2026-08-04T10:59:04+08:00 · 20260804_104428

<!-- experiment-record: {"run_dir":"outputs/ModelV1_dataset3_train_dataset4_val_V4/20260804_104428","completed_at":"2026-08-04T10:59:04+08:00","run_name":"20260804_104428","best_val_epe_mm":68.59001159667969,"best_epoch":52} -->

- 实验目的：Controlled camera/session holdout: train only on dataset 3 and validate only on dataset 4; otherwise identical to the RGB-face V4 point+pseudo-gaze baseline.
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_dataset3_train_dataset4_val_V4`，mode=`online`，tags=['ModelV1', 'V4', 'train_dataset_3', 'val_dataset_4', 'explicit_holdout', 'gaze_geometry', 'point_pnp', 'pseudo_gaze_cosine_100', 'rgb_face', 'imagenet_resnet18', 'eye_augmentation']
- 产物目录：`outputs/ModelV1_dataset3_train_dataset4_val_V4/20260804_104428`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=506，val=398，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 人脸图像骨干 | use=True，`resnet18`，weights=`DEFAULT`，freeze_until=`none` |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=22,784,478 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=0.0，gaze_angular_weight=0.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 52） | 41.968 | 68.590 mm | 64.393 mm | 25.218 mm | 60.202 mm |
| 最终训练（epoch 100） | 14.968 | 24.268 mm | 22.632 mm | 15.022 mm | 15.907 mm |
| 最终验证（epoch 100） | 43.213 | 70.028 mm | 65.539 mm | 20.233 mm | 64.106 mm |

相比上一实验：最佳 Val EPE 上升 5.198 mm，性能下降。

## 2026-08-04T23:19:12+08:00 · dataset5_v4_point_iris_ipd65_common971

<!-- experiment-record: {"run_dir":"outputs/ModelV1_dataset5_V4/dataset5_v4_point_iris_ipd65_common971","completed_at":"2026-08-04T23:19:12+08:00","run_name":"dataset5_v4_point_iris_ipd65_common971","best_val_epe_mm":41.2922477722168,"best_epoch":21} -->

- 实验目的：V4 stage 1: image predicts camera-space gaze; one PnP eye midpoint and parameter-free ray/table geometry predict UV.
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_dataset5_V4`，mode=`online`，tags=['ModelV1', 'V4', 'dataset_5', 'gaze_geometry', 'point_pnp', 'imagenet_resnet18', 'eye_augmentation']
- 产物目录：`outputs/ModelV1_dataset5_V4/dataset5_v4_point_iris_ipd65_common971`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `dataset_5`；train=782，val=189，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_iris_ipd_65mm_v1.csv` |
| 人脸图像骨干 | use=True，`resnet18`，weights=`DEFAULT`，freeze_until=`none` |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=22,784,091 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=1.0，gaze_angular_weight=0.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 21） | 15.428 | 41.292 mm | 32.598 mm | 22.042 mm | 31.029 mm |
| 最终训练（epoch 100） | 5.379 | 22.686 mm | 20.080 mm | 14.270 mm | 14.403 mm |
| 最终验证（epoch 100） | 20.161 | 51.244 mm | 47.343 mm | 32.559 mm | 32.430 mm |

相比上一实验：最佳 Val EPE 下降 27.298 mm，性能提升。

## 2026-08-05T08:55:02+08:00 · dataset3_4_v4_point_pnp1010_common842

<!-- experiment-record: {"run_dir":"outputs/ModelV1_dataset3_4_V4/dataset3_4_v4_point_pnp1010_common842","completed_at":"2026-08-05T08:55:02+08:00","run_name":"dataset3_4_v4_point_pnp1010_common842","best_val_epe_mm":71.78421783447266,"best_epoch":83} -->

- 实验目的：V4 stage 1: image predicts camera-space gaze; one PnP eye midpoint and parameter-free ray/table geometry predict UV.
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_dataset3_4_V4`，mode=`online`，tags=['ModelV1', 'V4', 'dataset_3_4', 'gaze_geometry', 'point_pnp', 'imagenet_resnet18', 'eye_augmentation']
- 产物目录：`outputs/ModelV1_dataset3_4_V4/dataset3_4_v4_point_pnp1010_common842`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=370，val=309，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 人脸图像骨干 | use=True，`resnet18`，weights=`DEFAULT`，freeze_until=`none` |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=22,784,091 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=1.0，gaze_angular_weight=0.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 83） | 31.389 | 71.784 mm | 68.760 mm | 24.702 mm | 63.418 mm |
| 最终训练（epoch 100） | 7.475 | 27.063 mm | 24.661 mm | 16.811 mm | 17.956 mm |
| 最终验证（epoch 100） | 32.058 | 73.098 mm | 73.128 mm | 25.380 mm | 64.029 mm |

相比上一实验：最佳 Val EPE 上升 30.492 mm，性能下降。

## 2026-08-05T09:07:34+08:00 · dataset3_4_v4_point_iris_ipd65_common842

<!-- experiment-record: {"run_dir":"outputs/ModelV1_dataset3_4_V4/dataset3_4_v4_point_iris_ipd65_common842","completed_at":"2026-08-05T09:07:34+08:00","run_name":"dataset3_4_v4_point_iris_ipd65_common842","best_val_epe_mm":69.06285095214844,"best_epoch":53} -->

- 实验目的：V4 stage 1: image predicts camera-space gaze; one PnP eye midpoint and parameter-free ray/table geometry predict UV.
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_dataset3_4_V4`，mode=`online`，tags=['ModelV1', 'V4', 'dataset_3_4', 'gaze_geometry', 'point_pnp', 'imagenet_resnet18', 'eye_augmentation']
- 产物目录：`outputs/ModelV1_dataset3_4_V4/dataset3_4_v4_point_iris_ipd65_common842`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=370，val=309，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_iris_ipd_65mm_v1.csv` |
| 人脸图像骨干 | use=True，`resnet18`，weights=`DEFAULT`，freeze_until=`none` |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=22,784,091 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=1.0，gaze_angular_weight=0.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 53） | 29.714 | 69.063 mm | 65.344 mm | 22.612 mm | 61.280 mm |
| 最终训练（epoch 100） | 7.126 | 26.256 mm | 24.731 mm | 16.450 mm | 17.502 mm |
| 最终验证（epoch 100） | 30.400 | 70.104 mm | 67.628 mm | 23.810 mm | 61.748 mm |

相比上一实验：最佳 Val EPE 下降 2.721 mm，性能提升。

## 2026-08-06T12:04:43+08:00 · numbered_train13_val3_v4_iris_ipd_65mm_fixed_prior_20260806_114607

<!-- experiment-record: {"run_dir":"outputs/ModelV1_numbered_V4/numbered_train13_val3_v4_iris_ipd_65mm_fixed_prior_20260806_114607","completed_at":"2026-08-06T12:04:43+08:00","run_name":"numbered_train13_val3_v4_iris_ipd_65mm_fixed_prior_20260806_114607","best_val_epe_mm":83.64015197753906,"best_epoch":30} -->

- 实验目的：Auto-generated numbered-artifact split: train=13, val=3, prior=iris_ipd_65mm, depth_mode=fixed_prior.
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_numbered_V4`，mode=`online`，tags=['ModelV1', 'v4', 'numbered_artifacts', 'gaze_geometry', 'iris_ipd_65mm', 'depth_fixed_prior', 'train_dataset_13', 'val_dataset_3', 'datasets_3_4_11_13', 'imagenet_resnet18', 'eye_augmentation']
- 产物目录：`outputs/ModelV1_numbered_V4/numbered_train13_val3_v4_iris_ipd_65mm_fixed_prior_20260806_114607`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=1047，val=305，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`numbered datasets [3, 4, 11, 13]; data/processed/deca_features_deca_crop_v1_<id>.npz` |
| 深度先验 | `numbered datasets [3, 4, 11, 13]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸图像骨干 | use=True，`resnet18`，weights=`DEFAULT`，freeze_until=`none` |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=22,784,091 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=1.0，gaze_angular_weight=0.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 30） | 39.958 | 83.640 mm | 78.504 mm | 63.520 mm | 42.964 mm |
| 最终训练（epoch 100） | 2.482 | 14.990 mm | 13.529 mm | 9.793 mm | 9.192 mm |
| 最终验证（epoch 100） | 44.940 | 91.296 mm | 89.783 mm | 71.639 mm | 45.296 mm |

相比上一实验：最佳 Val EPE 上升 14.577 mm，性能下降。

## 2026-08-07T09:42:52+08:00 · numbered_train3_val4_v4_iris_ipd_65mm_point_20260807_093016

<!-- experiment-record: {"run_dir":"outputs/ModelV1_numbered_V4/numbered_train3_val4_v4_iris_ipd_65mm_point_20260807_093016","completed_at":"2026-08-07T09:42:52+08:00","run_name":"numbered_train3_val4_v4_iris_ipd_65mm_point_20260807_093016","best_val_epe_mm":69.34169006347656,"best_epoch":68} -->

- 实验目的：Auto-generated numbered-artifact split: train=3, val=4, prior=iris_ipd_65mm, depth_mode=point.
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_numbered_V4`，mode=`online`，tags=['ModelV1', 'v4', 'numbered_artifacts', 'gaze_geometry', 'iris_ipd_65mm', 'depth_point', 'train_dataset_3', 'val_dataset_4', 'datasets_3_4_11_13', 'imagenet_resnet18', 'eye_augmentation']
- 产物目录：`outputs/ModelV1_numbered_V4/numbered_train3_val4_v4_iris_ipd_65mm_point_20260807_093016`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=310，val=277，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`numbered datasets [3, 4, 11, 13]; data/processed/deca_features_deca_crop_v1_<id>.npz` |
| 深度先验 | `numbered datasets [3, 4, 11, 13]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,263,739 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=1.0，gaze_angular_weight=0.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 68） | 30.117 | 69.342 mm | 66.234 mm | 24.599 mm | 60.729 mm |
| 最终训练（epoch 100） | 7.947 | 28.124 mm | 26.276 mm | 17.848 mm | 18.400 mm |
| 最终验证（epoch 100） | 31.173 | 71.210 mm | 66.199 mm | 24.929 mm | 62.103 mm |

相比上一实验：最佳 Val EPE 下降 14.298 mm，性能提升。

## 2026-08-07T11:42:28+08:00 · numbered_train3_val4_v4_iris_ipd_65mm_point_20260807_112933

<!-- experiment-record: {"run_dir":"outputs/ModelV1_numbered_V4/numbered_train3_val4_v4_iris_ipd_65mm_point_20260807_112933","completed_at":"2026-08-07T11:42:28+08:00","run_name":"numbered_train3_val4_v4_iris_ipd_65mm_point_20260807_112933","best_val_epe_mm":73.78179931640625,"best_epoch":56} -->

- 实验目的：Auto-generated numbered-artifact split: train=3, val=4, prior=iris_ipd_65mm, depth_mode=point.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_numbered_V4`，mode=`online`，tags=['ModelV1', 'v4', 'numbered_artifacts', 'gaze_geometry', 'iris_ipd_65mm', 'depth_point', 'train_dataset_3', 'val_dataset_4', 'datasets_3_4_11_13', 'imagenet_resnet18', 'eye_augmentation']
- 产物目录：`outputs/ModelV1_numbered_V4/numbered_train3_val4_v4_iris_ipd_65mm_point_20260807_112933`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=310，val=277，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [3, 4, 11, 13]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,156,803 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=1.0，gaze_angular_weight=0.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 56） | 32.157 | 73.782 mm | 69.730 mm | 21.772 mm | 67.516 mm |
| 最终训练（epoch 100） | 6.929 | 25.789 mm | 23.176 mm | 15.255 mm | 17.551 mm |
| 最终验证（epoch 100） | 33.557 | 77.798 mm | 77.226 mm | 18.651 mm | 72.689 mm |

相比上一实验：最佳 Val EPE 上升 4.440 mm，性能下降。

## 2026-08-07T15:31:36+08:00 · numbered_train3_val4_v4_iris_ipd_65mm_point_20260807_151924

<!-- experiment-record: {"run_dir":"outputs/ModelV1_numbered_V4/numbered_train3_val4_v4_iris_ipd_65mm_point_20260807_151924","completed_at":"2026-08-07T15:31:36+08:00","run_name":"numbered_train3_val4_v4_iris_ipd_65mm_point_20260807_151924","best_val_epe_mm":69.34169006347656,"best_epoch":68} -->

- 实验目的：Auto-generated numbered-artifact split: train=3, val=4, prior=iris_ipd_65mm, depth_mode=point.
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_numbered_V4`，mode=`online`，tags=['ModelV1', 'v4', 'numbered_artifacts', 'gaze_geometry', 'iris_ipd_65mm', 'depth_point', 'train_dataset_3', 'val_dataset_4', 'datasets_3_4_11_13', 'imagenet_resnet18', 'eye_augmentation']
- 产物目录：`outputs/ModelV1_numbered_V4/numbered_train3_val4_v4_iris_ipd_65mm_point_20260807_151924`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=310，val=277，seed=42 |
| DECA | representation=`geometry156`，preprocess=`deca`，crop_scale=1.25，cache=`numbered datasets [3, 4, 11, 13]; data/processed/deca_features_deca_crop_v1_<id>.npz` |
| 深度先验 | `numbered datasets [3, 4, 11, 13]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,263,739 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=1.0，gaze_angular_weight=0.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 68） | 30.117 | 69.342 mm | 66.234 mm | 24.599 mm | 60.729 mm |
| 最终训练（epoch 100） | 7.947 | 28.124 mm | 26.276 mm | 17.848 mm | 18.400 mm |
| 最终验证（epoch 100） | 31.173 | 71.210 mm | 66.199 mm | 24.929 mm | 62.103 mm |

相比上一实验：最佳 Val EPE 下降 4.440 mm，性能提升。

## 2026-08-07T23:50:26+08:00 · 20260807_233119

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260807_233119","completed_at":"2026-08-07T23:50:26+08:00","run_name":"20260807_233119","best_val_epe_mm":39.94624328613281,"best_epoch":58} -->

- 实验目的：Virtual-camera integration baseline: train dataset 3, validate unseen-camera dataset 4; predict gaze in N and rotate deterministically to C before geometry. DECA is disabled to isolate normalized RGB.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'V4', 'virtual_camera', 'gaze_N_to_C', 'train_dataset_3', 'val_dataset_4', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260807_233119`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=370，val=309，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `data/processed/modelv1_dataset_depth_prior_common.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,157,190 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=0.0，gaze_angular_weight=100.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 58） | 24.110 | 39.946 mm | 35.789 mm | 20.744 mm | 29.649 mm |
| 最终训练（epoch 100） | 14.578 | 23.086 mm | 19.999 mm | 13.127 mm | 15.925 mm |
| 最终验证（epoch 100） | 29.101 | 48.045 mm | 40.765 mm | 21.991 mm | 37.821 mm |

相比上一实验：最佳 Val EPE 下降 29.395 mm，性能提升。

## 2026-08-08T09:51:35+08:00 · numbered_train13_val3_v4_iris_ipd_65mm_point_20260808_092612

<!-- experiment-record: {"run_dir":"outputs/ModelV1_numbered_V4/numbered_train13_val3_v4_iris_ipd_65mm_point_20260808_092612","completed_at":"2026-08-08T09:51:35+08:00","run_name":"numbered_train13_val3_v4_iris_ipd_65mm_point_20260808_092612","best_val_epe_mm":83.23388671875,"best_epoch":39} -->

- 实验目的：Auto-generated numbered-artifact split: train=13, val=3, prior=iris_ipd_65mm, depth_mode=point.
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_numbered_V4`，mode=`online`，tags=['ModelV1', 'v4', 'numbered_artifacts', 'gaze_geometry', 'iris_ipd_65mm', 'depth_point', 'train_dataset_13', 'val_dataset_3', 'datasets_3_4_11_13', 'imagenet_resnet18', 'eye_augmentation']
- 产物目录：`outputs/ModelV1_numbered_V4/numbered_train13_val3_v4_iris_ipd_65mm_point_20260808_092612`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=1069，val=310，seed=42 |
| DECA | representation=`geometry156`，preprocess=`deca`，crop_scale=1.25，cache=`numbered datasets [3, 4, 11, 13]; data/processed/deca_features_deca_crop_v1_<id>.npz` |
| 深度先验 | `numbered datasets [3, 4, 11, 13]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,263,739 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=1.0，gaze_angular_weight=0.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 39） | 39.799 | 83.234 mm | 82.168 mm | 63.594 mm | 41.533 mm |
| 最终训练（epoch 100） | 1.887 | 13.008 mm | 11.783 mm | 8.281 mm | 8.238 mm |
| 最终验证（epoch 100） | 44.628 | 92.270 mm | 89.941 mm | 69.929 mm | 45.623 mm |

相比上一实验：最佳 Val EPE 上升 43.288 mm，性能下降。

## 2026-08-08T11:27:27+08:00 · 20260808_105208

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260808_105208","completed_at":"2026-08-08T11:27:27+08:00","run_name":"20260808_105208","best_val_epe_mm":106.19518280029297,"best_epoch":31} -->

- 实验目的：Virtual-camera integration baseline: train dataset 13, validate unseen-camera dataset 3; predict gaze in N and rotate deterministically to C before geometry. DECA is disabled to isolate normalized RGB.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'V4', 'virtual_camera', 'gaze_N_to_C', 'train_dataset_13', 'val_dataset_4', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260808_105208`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=1066，val=310，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [3, 13]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,157,190 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=0.0，gaze_angular_weight=100.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 31） | 80.654 | 106.195 mm | 104.974 mm | 64.268 mm | 71.203 mm |
| 最终训练（epoch 100） | 9.959 | 12.754 mm | 11.781 mm | 8.441 mm | 7.820 mm |
| 最终验证（epoch 100） | 98.839 | 124.722 mm | 122.931 mm | 63.203 mm | 95.347 mm |

相比上一实验：最佳 Val EPE 上升 22.961 mm，性能下降。

## 2026-08-08T12:16:37+08:00 · 20260808_114943

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260808_114943","completed_at":"2026-08-08T12:16:37+08:00","run_name":"20260808_114943","best_val_epe_mm":58.61956787109375,"best_epoch":28} -->

- 实验目的：Virtual-camera integration baseline: train dataset 11, validate unseen-camera dataset 4; predict gaze in N and rotate deterministically to C before geometry. DECA is disabled to isolate normalized RGB.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'V4', 'virtual_camera', 'gaze_N_to_C', 'train_dataset_11', 'val_dataset_4', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260808_114943`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=819，val=277，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [4, 11]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,157,190 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=0.0，gaze_angular_weight=100.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 28） | 38.172 | 58.620 mm | 53.381 mm | 36.908 mm | 38.645 mm |
| 最终训练（epoch 100） | 10.226 | 13.522 mm | 12.146 mm | 8.601 mm | 8.658 mm |
| 最终验证（epoch 100） | 50.009 | 73.105 mm | 67.556 mm | 37.167 mm | 54.523 mm |

相比上一实验：最佳 Val EPE 下降 47.576 mm，性能提升。

## 2026-08-08T21:07:45+08:00 · 20260808_203708

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260808_203708","completed_at":"2026-08-08T21:07:45+08:00","run_name":"20260808_203708","best_val_epe_mm":106.19518280029297,"best_epoch":31} -->

- 实验目的：Virtual-camera integration baseline: train dataset 13, validate unseen-camera dataset 3; predict gaze in N and rotate deterministically to C before geometry. DECA is disabled to isolate normalized RGB.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'V4', 'virtual_camera', 'gaze_N_to_C', 'train_dataset_13', 'val_dataset_3', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260808_203708`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=1066，val=310，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [3, 4, 11, 13]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,157,190 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=0.0，gaze_angular_weight=100.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 31） | 80.654 | 106.195 mm | 104.974 mm | 64.268 mm | 71.203 mm |
| 最终训练（epoch 100） | 9.959 | 12.754 mm | 11.781 mm | 8.441 mm | 7.820 mm |
| 最终验证（epoch 100） | 98.839 | 124.722 mm | 122.931 mm | 63.203 mm | 95.347 mm |

相比上一实验：最佳 Val EPE 上升 47.576 mm，性能下降。

## 2026-08-08T23:14:10+08:00 · 20260808_225452

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260808_225452","completed_at":"2026-08-08T23:14:10+08:00","run_name":"20260808_225452","best_val_epe_mm":79.56175231933594,"best_epoch":34} -->

- 实验目的：Virtual-camera integration baseline: train dataset 12, validate unseen-camera dataset 4; predict gaze in N and rotate deterministically to C before geometry. DECA is disabled to isolate normalized RGB.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'V4', 'virtual_camera', 'gaze_N_to_C', 'train_dataset_12', 'val_dataset_4', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260808_225452`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=572，val=277，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [3, 4, 5, 11, 12, 13]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,157,190 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=0.0，gaze_angular_weight=100.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 34） | 53.611 | 79.562 mm | 77.388 mm | 43.537 mm | 57.734 mm |
| 最终训练（epoch 100） | 10.670 | 14.947 mm | 13.487 mm | 10.191 mm | 8.743 mm |
| 最终验证（epoch 100） | 74.855 | 101.530 mm | 99.695 mm | 49.118 mm | 81.107 mm |

相比上一实验：最佳 Val EPE 下降 26.633 mm，性能提升。

## 2026-08-09T12:25:55+08:00 · 20260809_115900

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260809_115900","completed_at":"2026-08-09T12:25:55+08:00","run_name":"20260809_115900","best_val_epe_mm":94.3614273071289,"best_epoch":31} -->

- 实验目的：Virtual-camera integration baseline: train dataset 4, validate unseen-camera dataset 3; predict gaze in N and rotate deterministically to C before geometry. DECA is disabled to isolate normalized RGB.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'V4', 'virtual_camera', 'gaze_N_to_C', 'train_dataset_4', 'val_dataset_3', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260809_115900`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=1066，val=277，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [3, 4, 5, 11, 12, 13]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,157,190 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=0.0，gaze_angular_weight=100.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 31） | 68.157 | 94.361 mm | 90.270 mm | 48.752 mm | 72.339 mm |
| 最终训练（epoch 100） | 9.959 | 12.754 mm | 11.781 mm | 8.441 mm | 7.820 mm |
| 最终验证（epoch 100） | 89.137 | 117.683 mm | 116.134 mm | 47.872 mm | 100.255 mm |

相比上一实验：最佳 Val EPE 上升 14.800 mm，性能下降。

## 2026-08-09T17:16:54+08:00 · numbered_train4_val3_v4_iris_ipd_65mm_point_20260809_170619

<!-- experiment-record: {"run_dir":"outputs/ModelV1_numbered_V4/numbered_train4_val3_v4_iris_ipd_65mm_point_20260809_170619","completed_at":"2026-08-09T17:16:54+08:00","run_name":"numbered_train4_val3_v4_iris_ipd_65mm_point_20260809_170619","best_val_epe_mm":50.41913986206055,"best_epoch":35} -->

- 实验目的：Auto-generated numbered-artifact split: train=4, val=3, prior=iris_ipd_65mm, depth_mode=point.
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_numbered_V4`，mode=`online`，tags=['ModelV1', 'v4', 'numbered_artifacts', 'gaze_geometry', 'iris_ipd_65mm', 'depth_point', 'train_dataset_4', 'val_dataset_3', 'datasets_3_4_11_13', 'imagenet_resnet18', 'eye_augmentation']
- 产物目录：`outputs/ModelV1_numbered_V4/numbered_train4_val3_v4_iris_ipd_65mm_point_20260809_170619`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=277，val=310，seed=42 |
| DECA | representation=`geometry156`，preprocess=`deca`，crop_scale=1.25，cache=`numbered datasets [3, 4, 11, 13]; data/processed/deca_features_deca_crop_v1_<id>.npz` |
| 深度先验 | `numbered datasets [3, 4, 11, 13]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,263,739 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=1.0，gaze_angular_weight=0.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 35） | 20.029 | 50.419 mm | 46.718 mm | 30.706 mm | 33.537 mm |
| 最终训练（epoch 100） | 7.017 | 25.691 mm | 21.780 mm | 15.991 mm | 16.647 mm |
| 最终验证（epoch 100） | 23.670 | 57.044 mm | 55.907 mm | 31.316 mm | 40.365 mm |

相比上一实验：最佳 Val EPE 下降 43.942 mm，性能提升。

## 2026-08-09T19:23:51+08:00 · 20260809_185006

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260809_185006","completed_at":"2026-08-09T19:23:51+08:00","run_name":"20260809_185006","best_val_epe_mm":84.99526977539062,"best_epoch":39} -->

- 实验目的：Strict raw-image control for the paired Dataset13-to-Dataset4 virtual-camera experiment: predict gaze directly in camera coordinates while retaining the same sample IDs, Iris/IPD geometry, architecture, loss, and optimization settings.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'V4', 'legacy_control', 'gaze_camera', 'train_dataset_13', 'val_dataset_4', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260809_185006`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=1066，val=277，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [3, 4, 5, 11, 12, 13]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,157,190 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=0.0，gaze_angular_weight=100.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 39） | 59.090 | 84.995 mm | 84.489 mm | 42.953 mm | 65.645 mm |
| 最终训练（epoch 100） | 10.390 | 13.868 mm | 12.202 mm | 8.741 mm | 8.854 mm |
| 最终验证（epoch 100） | 73.564 | 100.006 mm | 100.696 mm | 45.196 mm | 81.030 mm |

相比上一实验：最佳 Val EPE 上升 34.576 mm，性能下降。

## 2026-08-09T20:44:13+08:00 · 20260809_200932

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260809_200932","completed_at":"2026-08-09T20:44:13+08:00","run_name":"20260809_200932","best_val_epe_mm":94.3614273071289,"best_epoch":31} -->

- 实验目的：Virtual-camera integration baseline: train dataset 13, validate unseen-camera dataset 4; predict gaze in N and rotate deterministically to C before geometry. DECA is disabled to isolate normalized RGB.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'V4', 'virtual_camera', 'gaze_N_to_C', 'train_dataset_14', 'val_dataset_4', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260809_200932`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=1066，val=277，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [3, 4, 5, 11, 12, 13]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,157,190 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=0.0，gaze_angular_weight=100.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 31） | 68.157 | 94.361 mm | 90.270 mm | 48.752 mm | 72.339 mm |
| 最终训练（epoch 100） | 9.959 | 12.754 mm | 11.781 mm | 8.441 mm | 7.820 mm |
| 最终验证（epoch 100） | 89.137 | 117.683 mm | 116.134 mm | 47.872 mm | 100.255 mm |

相比上一实验：最佳 Val EPE 上升 9.366 mm，性能下降。

## 2026-08-09T21:25:08+08:00 · 20260809_210949

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260809_210949","completed_at":"2026-08-09T21:25:08+08:00","run_name":"20260809_210949","best_val_epe_mm":73.14495086669922,"best_epoch":20} -->

- 实验目的：Strict raw-image control for the paired Dataset13-to-Dataset4 virtual-camera experiment: predict gaze directly in camera coordinates while retaining the same sample IDs, Iris/IPD geometry, architecture, loss, and optimization settings.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'V4', 'legacy_control', 'gaze_camera', 'train_dataset_3', 'val_dataset_4', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260809_210949`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=310，val=277，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [3, 4, 5, 11, 12, 13]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,157,190 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=0.0，gaze_angular_weight=100.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 20） | 43.389 | 73.145 mm | 69.596 mm | 24.415 mm | 65.919 mm |
| 最终训练（epoch 100） | 16.257 | 26.227 mm | 23.216 mm | 14.041 mm | 19.210 mm |
| 最终验证（epoch 100） | 51.928 | 83.460 mm | 87.459 mm | 21.037 mm | 77.456 mm |

相比上一实验：最佳 Val EPE 下降 21.216 mm，性能提升。

## 2026-08-09T23:04:35+08:00 · 20260809_222018

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260809_222018","completed_at":"2026-08-09T23:04:35+08:00","run_name":"20260809_222018","best_val_epe_mm":94.3614273071289,"best_epoch":31} -->

- 实验目的：Virtual-camera integration baseline: train dataset 13, validate unseen-camera dataset 4; predict gaze in N and rotate deterministically to C before geometry. DECA is disabled to isolate normalized RGB.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'V4', 'virtual_camera', 'gaze_N_to_C', 'train_dataset_14', 'val_dataset_4', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260809_222018`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=1066，val=277，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [3, 4, 5, 11, 12, 13]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,157,190 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=0.0，gaze_angular_weight=100.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 31） | 68.157 | 94.361 mm | 90.270 mm | 48.752 mm | 72.339 mm |
| 最终训练（epoch 100） | 9.959 | 12.754 mm | 11.781 mm | 8.441 mm | 7.820 mm |
| 最终验证（epoch 100） | 89.137 | 117.683 mm | 116.134 mm | 47.872 mm | 100.255 mm |

相比上一实验：最佳 Val EPE 上升 21.216 mm，性能下降。

## 2026-08-14T19:09:23+08:00 · 20260814_185141

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260814_185141","completed_at":"2026-08-14T19:09:23+08:00","run_name":"20260814_185141","best_val_epe_mm":93.58708953857422,"best_epoch":8} -->

- 实验目的：Preprocessed-v2 virtual-camera experiment: train Dataset 11, validate Dataset 3; select Iris65 or PnP1010 with data.depth_method, predict gaze in N, and rotate deterministically to C before geometry.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'V4', 'preprocessed_v2', 'virtual_camera', 'gaze_N_to_C', 'train_dataset_11', 'val_dataset_3', 'selectable_metric_depth', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260814_185141`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=384，val=262，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [11, 3]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,157,190 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=0.0，gaze_angular_weight=100.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 8） | 61.992 | 93.587 mm | 86.958 mm | 70.148 mm | 48.024 mm |
| 最终训练（epoch 100） | 12.319 | 18.892 mm | 17.578 mm | 12.307 mm | 11.802 mm |
| 最终验证（epoch 100） | 82.182 | 111.810 mm | 112.888 mm | 62.540 mm | 79.496 mm |

相比上一实验：最佳 Val EPE 下降 0.774 mm，性能提升。

## 2026-08-14T20:49:56+08:00 · 20260814_203134

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260814_203134","completed_at":"2026-08-14T20:49:56+08:00","run_name":"20260814_203134","best_val_epe_mm":93.39252471923828,"best_epoch":29} -->

- 实验目的：Preprocessed-v2 virtual-camera experiment: train Dataset 11, validate Dataset 3; select Iris65 or PnP1010 with data.depth_method, predict gaze in N, and rotate deterministically to C before geometry.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'V4', 'preprocessed_v2', 'virtual_camera', 'gaze_N_to_C', 'train_dataset_11', 'val_dataset_3', 'selectable_metric_depth', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260814_203134`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=384，val=262，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [11, 3]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,157,190 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=0.0，gaze_angular_weight=100.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 29） | 63.194 | 93.393 mm | 86.529 mm | 71.190 mm | 47.376 mm |
| 最终训练（epoch 100） | 12.298 | 18.832 mm | 17.260 mm | 12.197 mm | 11.920 mm |
| 最终验证（epoch 100） | 72.437 | 100.199 mm | 100.786 mm | 59.900 mm | 67.954 mm |

相比上一实验：最佳 Val EPE 下降 0.195 mm，性能提升。

## 2026-08-14T22:10:35+08:00 · 20260814_215245

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260814_215245","completed_at":"2026-08-14T22:10:35+08:00","run_name":"20260814_215245","best_val_epe_mm":91.92020416259766,"best_epoch":14} -->

- 实验目的：Preprocessed-v2 strict legacy-image control paired with the virtual-camera experiment: train Dataset 11, validate Dataset 3; use the same Iris65 depth contracts and valid sample IDs, but predict gaze directly in camera coordinates from the original images.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'V4', 'preprocessed_v2', 'legacy_control', 'gaze_camera', 'train_dataset_11', 'val_dataset_3', 'iris65', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260814_215245`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=384，val=262，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [11, 3]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,157,190 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=0.0，gaze_angular_weight=100.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 14） | 59.883 | 91.920 mm | 84.698 mm | 77.196 mm | 36.715 mm |
| 最终训练（epoch 100） | 13.759 | 21.438 mm | 18.894 mm | 14.379 mm | 13.142 mm |
| 最终验证（epoch 100） | 82.155 | 118.281 mm | 114.888 mm | 82.954 mm | 70.341 mm |

相比上一实验：最佳 Val EPE 下降 1.472 mm，性能提升。

## 2026-08-14T22:50:37+08:00 · 20260814_223316

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260814_223316","completed_at":"2026-08-14T22:50:37+08:00","run_name":"20260814_223316","best_val_epe_mm":91.92020416259766,"best_epoch":14} -->

- 实验目的：Preprocessed-v2 strict legacy-image control paired with the virtual-camera experiment: train Dataset 11, validate Dataset 3; use the same Iris65 depth contracts and valid sample IDs, but predict gaze directly in camera coordinates from the original images.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'V4', 'preprocessed_v2', 'legacy_control', 'gaze_camera', 'train_dataset_11', 'val_dataset_3', 'iris65', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260814_223316`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=384，val=262，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [11, 3]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,157,190 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=0.0，gaze_angular_weight=100.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 14） | 59.883 | 91.920 mm | 84.698 mm | 77.196 mm | 36.715 mm |
| 最终训练（epoch 100） | 13.759 | 21.438 mm | 18.894 mm | 14.379 mm | 13.142 mm |
| 最终验证（epoch 100） | 82.155 | 118.281 mm | 114.888 mm | 82.954 mm | 70.341 mm |

相比上一实验：最佳 Val EPE 与上一实验相同。

## 2026-08-14T23:20:20+08:00 · 20260814_230359

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260814_230359","completed_at":"2026-08-14T23:20:20+08:00","run_name":"20260814_230359","best_val_epe_mm":74.03887939453125,"best_epoch":23} -->

- 实验目的：Preprocessed-v2 strict legacy-image control paired with the virtual-camera experiment: train Dataset 11, validate Dataset 3; use the same Iris65 depth contracts and valid sample IDs, but predict gaze directly in camera coordinates from the original images.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'V4', 'preprocessed_v2', 'legacy_control', 'gaze_camera', 'train_dataset_11', 'val_dataset_3', 'iris65', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260814_230359`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=384，val=262，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [11, 3]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,157,190 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=0.0，gaze_angular_weight=100.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 23） | 49.336 | 74.039 mm | 62.192 mm | 57.995 mm | 36.275 mm |
| 最终训练（epoch 100） | 11.764 | 17.351 mm | 15.596 mm | 11.403 mm | 10.802 mm |
| 最终验证（epoch 100） | 54.647 | 76.502 mm | 68.826 mm | 60.197 mm | 38.382 mm |

相比上一实验：最佳 Val EPE 下降 17.881 mm，性能提升。

## 2026-08-16T00:01:55+08:00 · 20260815_234221

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260815_234221","completed_at":"2026-08-16T00:01:55+08:00","run_name":"20260815_234221","best_val_epe_mm":62.578765869140625,"best_epoch":21} -->

- 实验目的：Preprocessed-v2 direct-UV virtual-camera experiment: train Dataset 11, validate Dataset 3; transform the world gaze target through C into virtual camera N and supervise (x_N, y_N), encode TableFrame7 in N, and condition the learned UV fusion with FiLM from training-normalized log virtual-distance scale.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'uv_N', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'train_dataset_11', 'val_dataset_3', 'selectable_preprocessed_contract', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260815_234221`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=384，val=262，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [11, 3]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,200,304 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 21） | 0.370 | 62.579 mm | 57.894 mm | 50.515 mm | 28.057 mm |
| 最终训练（epoch 100） | 0.023 | 12.024 mm | 10.810 mm | 9.219 mm | 5.963 mm |
| 最终验证（epoch 100） | 0.447 | 72.181 mm | 64.129 mm | 55.837 mm | 33.199 mm |

相比上一实验：最佳 Val EPE 下降 11.460 mm，性能提升。

## 2026-08-16T22:33:19+08:00 · 20260816_220733

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260816_220733","completed_at":"2026-08-16T22:33:19+08:00","run_name":"20260816_220733","best_val_epe_mm":65.55160522460938,"best_epoch":26} -->

- 实验目的：Preprocessed-v2 direct-UV virtual-camera experiment: train Dataset 11, validate Dataset 3; transform the world gaze target through C into virtual camera N and supervise (x_N, y_N), encode TableFrame7 in N, and condition the learned UV fusion with FiLM from training-normalized log virtual-distance scale.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'uv_N', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'train_dataset_11', 'val_dataset_3', 'selectable_preprocessed_contract', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260816_220733`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=505，val=262，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [11, 3]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,200,304 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 26） | 0.352 | 65.552 mm | 59.755 mm | 53.680 mm | 29.316 mm |
| 最终训练（epoch 100） | 0.020 | 12.300 mm | 11.061 mm | 9.908 mm | 5.268 mm |
| 最终验证（epoch 100） | 0.448 | 71.940 mm | 67.849 mm | 51.725 mm | 40.679 mm |

相比上一实验：最佳 Val EPE 上升 2.973 mm，性能下降。

## 2026-08-16T23:21:48+08:00 · 20260816_224737

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260816_224737","completed_at":"2026-08-16T23:21:48+08:00","run_name":"20260816_224737","best_val_epe_mm":65.44842529296875,"best_epoch":26} -->

- 实验目的：Preprocessed-v2 direct-UV virtual-camera experiment: train Dataset 11, validate Dataset 3; transform the world gaze target through C into virtual camera N and supervise (x_N, y_N), encode TableFrame7 in N, and condition the learned UV fusion with FiLM from training-normalized log virtual-distance scale.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'uv_N', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'train_dataset_11', 'val_dataset_3', 'selectable_preprocessed_contract', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260816_224737`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=505，val=319，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [11, 3]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,200,304 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 26） | 0.352 | 65.448 mm | 59.518 mm | 53.768 mm | 29.162 mm |
| 最终训练（epoch 100） | 0.020 | 12.300 mm | 11.061 mm | 9.908 mm | 5.268 mm |
| 最终验证（epoch 100） | 0.451 | 72.159 mm | 66.836 mm | 52.084 mm | 40.894 mm |

相比上一实验：最佳 Val EPE 下降 0.103 mm，性能提升。

## 2026-08-17T00:13:06+08:00 · 20260816_234921

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260816_234921","completed_at":"2026-08-17T00:13:06+08:00","run_name":"20260816_234921","best_val_epe_mm":65.44842529296875,"best_epoch":26} -->

- 实验目的：Preprocessed-v2 direct-UV virtual-camera experiment: train Dataset 11, validate Dataset 3; transform the world gaze target through C into virtual camera N and supervise (x_N, y_N), encode TableFrame7 in N, and condition the learned UV fusion with FiLM from training-normalized log virtual-distance scale.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'uv_N', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'train_dataset_11', 'val_dataset_3', 'selectable_preprocessed_contract', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260816_234921`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=505，val=319，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [11, 3]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,200,304 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 26） | 0.352 | 65.448 mm | 59.518 mm | 53.768 mm | 29.162 mm |
| 最终训练（epoch 100） | 0.020 | 12.300 mm | 11.061 mm | 9.908 mm | 5.268 mm |
| 最终验证（epoch 100） | 0.451 | 72.159 mm | 66.836 mm | 52.084 mm | 40.894 mm |

相比上一实验：最佳 Val EPE 与上一实验相同。

## 2026-08-17T10:55:00+08:00 · 20260817_101832

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260817_101832","completed_at":"2026-08-17T10:55:00+08:00","run_name":"20260817_101832","best_val_epe_mm":29.672170639038086,"best_epoch":20} -->

- 实验目的：Preprocessed-v2 direct-UV virtual-camera experiment: train Dataset 11, validate Dataset 3; transform the world gaze target through C into virtual camera N and supervise (x_N, y_N), encode TableFrame7 in N, and condition the learned UV fusion with FiLM from training-normalized log virtual-distance scale.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'uv_N', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'train_dataset_11', 'val_dataset_3', 'selectable_preprocessed_contract', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260817_101832`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=801，val=505，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [11, 3, 13]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,200,304 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 20） | 0.123 | 29.672 mm | 27.041 mm | 21.538 mm | 15.846 mm |
| 最终训练（epoch 100） | 0.015 | 10.711 mm | 9.374 mm | 9.255 mm | 3.870 mm |
| 最终验证（epoch 100） | 0.192 | 33.666 mm | 33.547 mm | 20.381 mm | 23.161 mm |

相比上一实验：最佳 Val EPE 下降 35.776 mm，性能提升。

## 2026-08-17T15:22:14+08:00 · 20260817_150211

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260817_150211","completed_at":"2026-08-17T15:22:14+08:00","run_name":"20260817_150211","best_val_epe_mm":28.11717414855957,"best_epoch":20} -->

- 实验目的：Preprocessed-v2 direct-UV virtual-camera experiment: train Dataset 11, validate Dataset 3; transform the world gaze target through C into virtual camera N and supervise (x_N, y_N), encode TableFrame7 in N, and condition the learned UV fusion with FiLM from training-normalized log virtual-distance scale.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'uv_N', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'train_dataset_11', 'val_dataset_3', 'selectable_preprocessed_contract', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260817_150211`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=801，val=505，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [11, 3, 13]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,200,304 |
| 训练 | epochs=50，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 20） | 0.114 | 28.117 mm | 26.450 mm | 21.152 mm | 14.633 mm |
| 最终训练（epoch 50） | 0.023 | 13.377 mm | 11.298 mm | 11.595 mm | 4.692 mm |
| 最终验证（epoch 50） | 0.162 | 32.072 mm | 31.188 mm | 21.231 mm | 20.234 mm |

相比上一实验：最佳 Val EPE 下降 1.555 mm，性能提升。

## 2026-08-17T15:39:55+08:00 · 20260817_152440

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260817_152440","completed_at":"2026-08-17T15:39:55+08:00","run_name":"20260817_152440","best_val_epe_mm":27.44663429260254,"best_epoch":31} -->

- 实验目的：Preprocessed-v2 direct-UV virtual-camera experiment: train Dataset 11, validate Dataset 3; transform the world gaze target through C into virtual camera N and supervise (x_N, y_N), encode TableFrame7 in N, and condition the learned UV fusion with FiLM from training-normalized log virtual-distance scale.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'uv_N', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'train_dataset_11', 'val_dataset_3', 'selectable_preprocessed_contract', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260817_152440`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=801，val=505，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [11, 3, 13]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,200,304 |
| 训练 | epochs=50，batch=64，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 31） | 0.096 | 27.447 mm | 25.285 mm | 22.704 mm | 12.344 mm |
| 最终训练（epoch 50） | 0.034 | 16.261 mm | 14.029 mm | 14.154 mm | 5.676 mm |
| 最终验证（epoch 50） | 0.112 | 28.746 mm | 25.571 mm | 21.754 mm | 14.841 mm |

相比上一实验：最佳 Val EPE 下降 0.671 mm，性能提升。

## 2026-08-17T16:07:15+08:00 · 20260817_155210

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260817_155210","completed_at":"2026-08-17T16:07:15+08:00","run_name":"20260817_155210","best_val_epe_mm":26.698780059814453,"best_epoch":36} -->

- 实验目的：Preprocessed-v2 direct-UV virtual-camera experiment: train Dataset 11, validate Dataset 3; transform the world gaze target through C into virtual camera N and supervise (x_N, y_N), encode TableFrame7 in N, and condition the learned UV fusion with FiLM from training-normalized log virtual-distance scale.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'uv_N', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'train_dataset_11', 'val_dataset_3', 'selectable_preprocessed_contract', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260817_155210`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=801，val=505，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [11, 3, 13]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,200,304 |
| 训练 | epochs=50，batch=64，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 36） | 0.093 | 26.699 mm | 23.175 mm | 21.438 mm | 12.033 mm |
| 最终训练（epoch 50） | 0.035 | 16.485 mm | 14.108 mm | 14.366 mm | 5.691 mm |
| 最终验证（epoch 50） | 0.107 | 27.889 mm | 24.658 mm | 21.009 mm | 14.456 mm |

相比上一实验：最佳 Val EPE 下降 0.748 mm，性能提升。

## 2026-08-17T16:36:11+08:00 · 20260817_162644

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260817_162644","completed_at":"2026-08-17T16:36:11+08:00","run_name":"20260817_162644","best_val_epe_mm":28.987510681152344,"best_epoch":50} -->

- 实验目的：Preprocessed-v2 direct-UV virtual-camera experiment: train Dataset 11, validate Dataset 3; transform the world gaze target through C into virtual camera N and supervise (x_N, y_N), encode TableFrame7 in N, and condition the learned UV fusion with FiLM from training-normalized log virtual-distance scale.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'uv_N', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'train_dataset_11', 'val_dataset_3', 'selectable_preprocessed_contract', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260817_162644`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=801，val=505，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [11, 3, 13]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=True |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,200,304 |
| 训练 | epochs=50，batch=64，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 50） | 0.110 | 28.988 mm | 25.522 mm | 23.653 mm | 12.633 mm |
| 最终训练（epoch 50） | 0.045 | 18.672 mm | 16.315 mm | 15.917 mm | 6.923 mm |
| 最终验证（epoch 50） | 0.110 | 28.988 mm | 25.522 mm | 23.653 mm | 12.633 mm |

相比上一实验：最佳 Val EPE 上升 2.289 mm，性能下降。

## 2026-08-17T18:30:22+08:00 · 20260817_181332

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260817_181332","completed_at":"2026-08-17T18:30:22+08:00","run_name":"20260817_181332","best_val_epe_mm":30.241151809692383,"best_epoch":34} -->

- 实验目的：Preprocessed-v2 direct-UV virtual-camera experiment: train Dataset 11, validate Dataset 3; transform the world gaze target through C into virtual camera N and supervise (x_N, y_N), encode TableFrame7 in N, and condition the learned UV fusion with FiLM from training-normalized log virtual-distance scale.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'uv_N', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'train_dataset_11', 'val_dataset_3', 'selectable_preprocessed_contract', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260817_181332`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=801，val=505，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [11, 3, 13]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,200,304 |
| 训练 | epochs=50，batch=64，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 34） | 0.123 | 30.241 mm | 27.999 mm | 23.500 mm | 15.184 mm |
| 最终训练（epoch 50） | 0.037 | 17.218 mm | 14.760 mm | 15.082 mm | 5.850 mm |
| 最终验证（epoch 50） | 0.151 | 32.188 mm | 30.472 mm | 22.911 mm | 18.641 mm |

相比上一实验：最佳 Val EPE 上升 1.254 mm，性能下降。

## 2026-08-17T23:23:26+08:00 · 20260817_230855

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260817_230855","completed_at":"2026-08-17T23:23:26+08:00","run_name":"20260817_230855","best_val_epe_mm":29.252960205078125,"best_epoch":29} -->

- 实验目的：Preprocessed-v2 direct-UV virtual-camera experiment: train Dataset 11, validate Dataset 3; transform the world gaze target through C into virtual camera N and supervise (x_N, y_N), encode TableFrame7 in N, and condition the learned UV fusion with FiLM from training-normalized log virtual-distance scale.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'uv_N', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'train_dataset_11', 'val_dataset_3', 'selectable_preprocessed_contract', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260817_230855`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=801，val=505，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [11, 3, 13]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,200,304 |
| 训练 | epochs=50，batch=64，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 29） | 0.122 | 29.253 mm | 27.637 mm | 21.794 mm | 15.580 mm |
| 最终训练（epoch 50） | 0.036 | 16.862 mm | 14.392 mm | 14.752 mm | 5.677 mm |
| 最终验证（epoch 50） | 0.132 | 30.117 mm | 28.070 mm | 21.457 mm | 17.121 mm |

相比上一实验：最佳 Val EPE 下降 0.988 mm，性能提升。

## 2026-08-18T09:30:20+08:00 · 20260818_091627

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260818_091627","completed_at":"2026-08-18T09:30:20+08:00","run_name":"20260818_091627","best_val_epe_mm":34.990966796875,"best_epoch":36} -->

- 实验目的：Paired virtual-camera gaze-geometry control for the direct-UV experiment: train Dataset 13, validate Dataset 11; use the same Iris65 preprocessed-v2 contract, virtual images, visual backbone, augmentation, batch size, optimizer, schedule, and epoch budget. The only prediction-path difference is gaze_N -> deterministic N-to-C rotation -> ray/table geometry.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'V4', 'preprocessed_v2', 'virtual_camera', 'gaze_N_to_C', 'paired_geometry_control', 'iris65', 'train_dataset_13', 'val_dataset_11', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260818_091627`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=801，val=505，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [11, 3, 13]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,157,190 |
| 训练 | epochs=50，batch=64，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=0.0，gaze_angular_weight=100.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 36） | 21.210 | 34.991 mm | 32.935 mm | 24.832 mm | 20.182 mm |
| 最终训练（epoch 50） | 14.479 | 23.252 mm | 21.116 mm | 15.584 mm | 14.094 mm |
| 最终验证（epoch 50） | 21.442 | 35.267 mm | 33.308 mm | 24.687 mm | 20.577 mm |

相比上一实验：最佳 Val EPE 上升 5.738 mm，性能下降。

## 2026-08-18T10:02:19+08:00 · 20260818_094749

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260818_094749","completed_at":"2026-08-18T10:02:19+08:00","run_name":"20260818_094749","best_val_epe_mm":41.26337814331055,"best_epoch":29} -->

- 实验目的：Preprocessed-v2 direct-UV virtual-camera experiment: output normalized P_N[:2], reconstruct P_N.z on the known per-sample table plane, and compute loss/EPE against table-local UV in millimetres; encode TableFrame7 in N and condition fusion with FiLM from training-normalized log virtual-distance scale.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'uv_N_output_table_UV_loss', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'train_dataset_11', 'val_dataset_3', 'selectable_preprocessed_contract', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260818_094749`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=801，val=505，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [11, 3, 13]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,200,304 |
| 训练 | epochs=50，batch=64，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 29） | 15.032 | 41.263 mm | 39.759 mm | 28.693 mm | 24.421 mm |
| 最终训练（epoch 50） | 4.484 | 20.544 mm | 18.583 mm | 11.598 mm | 14.490 mm |
| 最终验证（epoch 50） | 20.006 | 50.774 mm | 50.847 mm | 42.092 mm | 21.794 mm |

相比上一实验：最佳 Val EPE 上升 6.272 mm，性能下降。

## 2026-08-18T10:58:09+08:00 · 20260818_104409

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260818_104409","completed_at":"2026-08-18T10:58:09+08:00","run_name":"20260818_104409","best_val_epe_mm":36.93905258178711,"best_epoch":30} -->

- 实验目的：Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from training-normalized log virtual-distance scale.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'direct_table_UV', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'train_dataset_13', 'val_dataset_11', 'selectable_preprocessed_contract', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260818_104409`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=801，val=505，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [11, 3, 13]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,200,304 |
| 训练 | epochs=50，batch=64，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 30） | 12.646 | 36.939 mm | 32.962 mm | 22.620 mm | 24.748 mm |
| 最终训练（epoch 50） | 5.050 | 21.989 mm | 20.415 mm | 13.902 mm | 14.226 mm |
| 最终验证（epoch 50） | 13.919 | 39.452 mm | 38.124 mm | 26.805 mm | 24.140 mm |

相比上一实验：最佳 Val EPE 下降 4.324 mm，性能提升。

## 2026-08-18T15:33:01+08:00 · 20260818_151611

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260818_151611","completed_at":"2026-08-18T15:33:01+08:00","run_name":"20260818_151611","best_val_epe_mm":33.97908401489258,"best_epoch":25} -->

- 实验目的：Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from Condition17=[TableFrame7_N, log virtual-distance scale, virtual camera pose in table frame].
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'direct_table_UV', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'virtual_pose_FiLM', 'Condition17', 'train_dataset_13', 'val_dataset_11', 'selectable_preprocessed_contract', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260818_151611`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=801，val=505，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [11, 3, 13]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,200,594 |
| 训练 | epochs=50，batch=64，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 25） | 10.866 | 33.979 mm | 32.199 mm | 21.486 mm | 22.208 mm |
| 最终训练（epoch 50） | 5.298 | 22.518 mm | 20.345 mm | 13.094 mm | 15.472 mm |
| 最终验证（epoch 50） | 14.465 | 40.293 mm | 37.045 mm | 27.745 mm | 23.981 mm |

相比上一实验：最佳 Val EPE 下降 2.960 mm，性能提升。

## 2026-08-18T17:05:35+08:00 · 20260818_165029

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260818_165029","completed_at":"2026-08-18T17:05:35+08:00","run_name":"20260818_165029","best_val_epe_mm":42.01571273803711,"best_epoch":8} -->

- 实验目的：Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from Condition17=[TableFrame7_N, log virtual-distance scale, virtual camera pose in table frame].
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'direct_table_UV', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'virtual_pose_FiLM', 'Condition17', 'train_dataset_13', 'val_dataset_11', 'selectable_preprocessed_contract', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260818_165029`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=801，val=505，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [11, 3, 13]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,200,450 |
| 训练 | epochs=50，batch=64，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 8） | 15.200 | 42.016 mm | 38.985 mm | 23.387 mm | 30.119 mm |
| 最终训练（epoch 50） | 5.377 | 22.694 mm | 21.260 mm | 13.513 mm | 15.252 mm |
| 最终验证（epoch 50） | 20.393 | 51.442 mm | 48.938 mm | 41.893 mm | 22.901 mm |

相比上一实验：最佳 Val EPE 上升 8.037 mm，性能下降。

## 2026-08-18T23:17:08+08:00 · 20260818_230245

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260818_230245","completed_at":"2026-08-18T23:17:08+08:00","run_name":"20260818_230245","best_val_epe_mm":39.788570404052734,"best_epoch":19} -->

- 实验目的：Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from Condition17=[TableFrame7_N, log virtual-distance scale, virtual camera pose in table frame].
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'direct_table_UV', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'virtual_pose_FiLM', 'Condition17', 'train_dataset_13', 'val_dataset_11', 'selectable_preprocessed_contract', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260818_230245`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=801，val=505，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [11, 3, 13]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,200,450 |
| 训练 | epochs=50，batch=64，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 19） | 0.133 | 39.789 mm | 35.948 mm | 27.845 mm | 23.016 mm |
| 最终训练（epoch 50） | 0.052 | 23.291 mm | 21.010 mm | 13.258 mm | 16.325 mm |
| 最终验证（epoch 50） | 0.204 | 52.181 mm | 50.092 mm | 42.673 mm | 22.989 mm |

相比上一实验：最佳 Val EPE 下降 2.227 mm，性能提升。

## 2026-08-19T15:50:46+08:00 · 20260819_153332

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260819_153332","completed_at":"2026-08-19T15:50:46+08:00","run_name":"20260819_153332","best_val_epe_mm":34.38920974731445,"best_epoch":25} -->

- 实验目的：Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from Condition17=[TableFrame7_N, log virtual-distance scale, virtual camera pose in table frame].
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'direct_table_UV', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'virtual_pose_FiLM', 'Condition17', 'train_dataset_13', 'val_dataset_11', 'selectable_preprocessed_contract', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260819_153332`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=798，val=505，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [13, 11]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,200,450 |
| 训练 | epochs=50，batch=64，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 25） | 10.928 | 34.389 mm | 32.021 mm | 21.202 mm | 22.244 mm |
| 最终训练（epoch 50） | 5.342 | 22.591 mm | 20.762 mm | 14.176 mm | 14.531 mm |
| 最终验证（epoch 50） | 12.382 | 36.769 mm | 33.230 mm | 23.328 mm | 23.679 mm |

相比上一实验：最佳 Val EPE 下降 5.399 mm，性能提升。

## 2026-08-19T17:12:04+08:00 · 20260819_165805

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260819_165805","completed_at":"2026-08-19T17:12:04+08:00","run_name":"20260819_165805","best_val_epe_mm":32.22655487060547,"best_epoch":25} -->

- 实验目的：Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy eye images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from Condition17=[TableFrame7_N, log virtual-distance scale, virtual camera pose in table frame].
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'direct_table_UV', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'virtual_pose_FiLM', 'Condition17', 'train_dataset_13', 'val_dataset_11', 'selectable_preprocessed_contract', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260819_165805`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=798，val=505，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [13, 11]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,200,450 |
| 训练 | epochs=50，batch=64，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 25） | 9.836 | 32.227 mm | 30.124 mm | 20.161 mm | 21.171 mm |
| 最终训练（epoch 50） | 4.954 | 21.798 mm | 19.833 mm | 13.854 mm | 14.038 mm |
| 最终验证（epoch 50） | 11.337 | 34.742 mm | 32.519 mm | 20.756 mm | 23.548 mm |

相比上一实验：最佳 Val EPE 下降 2.163 mm，性能提升。

## 2026-08-20T09:14:55+08:00 · 20260820_085926

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260820_085926","completed_at":"2026-08-20T09:14:55+08:00","run_name":"20260820_085926","best_val_epe_mm":34.23834991455078,"best_epoch":18} -->

- 实验目的：Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy eye images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from Condition17=[TableFrame7_N, log virtual-distance scale, virtual camera pose in table frame].
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'direct_table_UV', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'virtual_pose_FiLM', 'Condition17', 'train_dataset_13', 'val_dataset_11', 'selectable_preprocessed_contract', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260820_085926`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=798，val=505，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [13, 11]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，input_size=`(60, 36)` (W,H)，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,200,450 |
| 训练 | epochs=50，batch=64，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 18） | 11.039 | 34.238 mm | 31.743 mm | 18.966 mm | 24.284 mm |
| 最终训练（epoch 50） | 5.345 | 22.642 mm | 20.979 mm | 14.179 mm | 14.672 mm |
| 最终验证（epoch 50） | 11.939 | 35.895 mm | 33.011 mm | 21.773 mm | 24.008 mm |

相比上一实验：最佳 Val EPE 上升 2.012 mm，性能下降。

## 2026-08-20T10:21:55+08:00 · 20260820_100534

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260820_100534","completed_at":"2026-08-20T10:21:55+08:00","run_name":"20260820_100534","best_val_epe_mm":33.18215560913086,"best_epoch":45} -->

- 实验目的：Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy eye images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from Condition17=[TableFrame7_N, log virtual-distance scale, virtual camera pose in table frame].
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'direct_table_UV', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'virtual_pose_FiLM', 'Condition17', 'train_dataset_13', 'val_dataset_11', 'selectable_preprocessed_contract', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260820_100534`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=798，val=505，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [13, 11]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，input_size=`(60, 36)` (W,H)，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,200,450 |
| 训练 | epochs=50，batch=64，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 45） | 10.534 | 33.182 mm | 30.337 mm | 19.459 mm | 22.931 mm |
| 最终训练（epoch 50） | 5.280 | 22.433 mm | 20.373 mm | 14.045 mm | 14.605 mm |
| 最终验证（epoch 50） | 11.307 | 34.668 mm | 31.608 mm | 21.031 mm | 23.166 mm |

相比上一实验：最佳 Val EPE 下降 1.056 mm，性能提升。

## 2026-08-20T11:03:59+08:00 · 20260820_104613

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260820_104613","completed_at":"2026-08-20T11:03:59+08:00","run_name":"20260820_104613","best_val_epe_mm":33.18215560913086,"best_epoch":45} -->

- 实验目的：Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy eye images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from Condition17=[TableFrame7_N, log virtual-distance scale, virtual camera pose in table frame].
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'direct_table_UV', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'virtual_pose_FiLM', 'Condition17', 'train_dataset_13', 'val_dataset_11', 'selectable_preprocessed_contract', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260820_104613`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=798，val=505，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [13, 11]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，input_size=`(60, 36)` (W,H)，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,200,450 |
| 训练 | epochs=50，batch=64，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 45） | 10.534 | 33.182 mm | 30.337 mm | 19.459 mm | 22.931 mm |
| 最终训练（epoch 50） | 5.280 | 22.433 mm | 20.373 mm | 14.045 mm | 14.605 mm |
| 最终验证（epoch 50） | 11.307 | 34.668 mm | 31.608 mm | 21.031 mm | 23.166 mm |

相比上一实验：最佳 Val EPE 与上一实验相同。

## 2026-08-20T11:39:34+08:00 · 20260820_112502

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260820_112502","completed_at":"2026-08-20T11:39:34+08:00","run_name":"20260820_112502","best_val_epe_mm":33.18215560913086,"best_epoch":45} -->

- 实验目的：Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy eye images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from Condition17=[TableFrame7_N, log virtual-distance scale, virtual camera pose in table frame].
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'direct_table_UV', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'virtual_pose_FiLM', 'Condition17', 'train_dataset_13', 'val_dataset_11', 'selectable_preprocessed_contract', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260820_112502`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=798，val=505，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [13, 11]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，input_size=`(60, 36)` (W,H)，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,200,450 |
| 训练 | epochs=50，batch=64，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 45） | 10.534 | 33.182 mm | 30.337 mm | 19.459 mm | 22.931 mm |
| 最终训练（epoch 50） | 5.280 | 22.433 mm | 20.373 mm | 14.045 mm | 14.605 mm |
| 最终验证（epoch 50） | 11.307 | 34.668 mm | 31.608 mm | 21.031 mm | 23.166 mm |

相比上一实验：最佳 Val EPE 与上一实验相同。

## 2026-08-20T15:20:42+08:00 · 20260820_150433

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260820_150433","completed_at":"2026-08-20T15:20:42+08:00","run_name":"20260820_150433","best_val_epe_mm":34.481109619140625,"best_epoch":19} -->

- 实验目的：Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy eye images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from Condition17=[TableFrame7_N, log virtual-distance scale, virtual camera pose in table frame].
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'direct_table_UV', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'virtual_pose_FiLM', 'Condition17', 'train_dataset_13', 'val_dataset_11', 'selectable_preprocessed_contract', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260820_150433`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=798，val=505，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [13, 11]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，input_size=`(60, 36)` (W,H)，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,200,962 |
| 训练 | epochs=50，batch=64，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 19） | 10.866 | 34.481 mm | 32.337 mm | 19.999 mm | 23.813 mm |
| 最终训练（epoch 50） | 5.233 | 22.594 mm | 21.152 mm | 13.454 mm | 15.336 mm |
| 最终验证（epoch 50） | 11.738 | 35.647 mm | 32.281 mm | 20.735 mm | 24.470 mm |

相比上一实验：最佳 Val EPE 上升 1.299 mm，性能下降。

## 2026-08-20T16:03:25+08:00 · 20260820_154852

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260820_154852","completed_at":"2026-08-20T16:03:25+08:00","run_name":"20260820_154852","best_val_epe_mm":34.00495910644531,"best_epoch":38} -->

- 实验目的：Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy eye images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from Condition17=[TableFrame7_N, log virtual-distance scale, virtual camera pose in table frame].
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'direct_table_UV', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'virtual_pose_FiLM', 'Condition17', 'train_dataset_13', 'val_dataset_11', 'selectable_preprocessed_contract', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260820_154852`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=798，val=505，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [13, 11]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，input_size=`(60, 36)` (W,H)，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,172,802 |
| 训练 | epochs=50，batch=64，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 38） | 10.926 | 34.005 mm | 31.995 mm | 19.816 mm | 23.675 mm |
| 最终训练（epoch 50） | 4.584 | 20.861 mm | 19.233 mm | 12.761 mm | 13.734 mm |
| 最终验证（epoch 50） | 11.818 | 35.485 mm | 32.941 mm | 20.739 mm | 24.584 mm |

相比上一实验：最佳 Val EPE 下降 0.476 mm，性能提升。

## 2026-08-20T16:35:09+08:00 · 20260820_161934

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260820_161934","completed_at":"2026-08-20T16:35:09+08:00","run_name":"20260820_161934","best_val_epe_mm":32.78703308105469,"best_epoch":30} -->

- 实验目的：Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy eye images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from Condition17=[TableFrame7_N, log virtual-distance scale, virtual camera pose in table frame].
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'direct_table_UV', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'virtual_pose_FiLM', 'Condition17', 'train_dataset_13', 'val_dataset_11', 'selectable_preprocessed_contract', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260820_161934`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=798，val=505，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [13, 11]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，input_size=`[90, 56]` (W,H)，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,172,802 |
| 训练 | epochs=50，batch=64，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 30） | 10.195 | 32.787 mm | 30.134 mm | 18.474 mm | 23.035 mm |
| 最终训练（epoch 50） | 4.751 | 21.220 mm | 19.340 mm | 13.614 mm | 13.461 mm |
| 最终验证（epoch 50） | 10.811 | 33.668 mm | 31.607 mm | 19.524 mm | 23.413 mm |

相比上一实验：最佳 Val EPE 下降 1.218 mm，性能提升。

## 2026-08-20T19:09:27+08:00 · 20260820_185238

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260820_185238","completed_at":"2026-08-20T19:09:27+08:00","run_name":"20260820_185238","best_val_epe_mm":33.34921646118164,"best_epoch":41} -->

- 实验目的：Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy eye images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from Condition17=[TableFrame7_N, log virtual-distance scale, virtual camera pose in table frame].
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'direct_table_UV', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'virtual_pose_FiLM', 'Condition17', 'train_dataset_13', 'val_dataset_11', 'selectable_preprocessed_contract', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260820_185238`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=798，val=505，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [13, 11]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，input_size=`[90, 56]` (W,H)，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,845,570 |
| 训练 | epochs=50，batch=64，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 41） | 10.563 | 33.349 mm | 29.868 mm | 20.045 mm | 22.789 mm |
| 最终训练（epoch 50） | 5.224 | 22.312 mm | 20.872 mm | 13.892 mm | 14.462 mm |
| 最终验证（epoch 50） | 10.855 | 33.824 mm | 30.281 mm | 19.880 mm | 23.163 mm |

相比上一实验：最佳 Val EPE 上升 0.562 mm，性能下降。

## 2026-08-20T20:25:14+08:00 · 20260820_200821

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260820_200821","completed_at":"2026-08-20T20:25:14+08:00","run_name":"20260820_200821","best_val_epe_mm":33.40460968017578,"best_epoch":41} -->

- 实验目的：Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy eye images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from Condition17=[TableFrame7_N, log virtual-distance scale, virtual camera pose in table frame].
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'direct_table_UV', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'virtual_pose_FiLM', 'Condition17', 'train_dataset_13', 'val_dataset_11', 'selectable_preprocessed_contract', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260820_200821`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=798，val=505，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [13, 11]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，input_size=`[90, 56]` (W,H)，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,845,570 |
| 训练 | epochs=50，batch=64，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 41） | 10.584 | 33.405 mm | 29.930 mm | 20.042 mm | 22.817 mm |
| 最终训练（epoch 50） | 5.214 | 22.270 mm | 20.542 mm | 13.898 mm | 14.401 mm |
| 最终验证（epoch 50） | 10.827 | 33.808 mm | 29.996 mm | 19.854 mm | 23.143 mm |

相比上一实验：最佳 Val EPE 上升 0.055 mm，性能下降。

## 2026-08-20T22:18:58+08:00 · 20260820_220340

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260820_220340","completed_at":"2026-08-20T22:18:58+08:00","run_name":"20260820_220340","best_val_epe_mm":33.284358978271484,"best_epoch":41} -->

- 实验目的：Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy eye images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from Condition17=[TableFrame7_N, log virtual-distance scale, virtual camera pose in table frame].
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'direct_table_UV', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'virtual_pose_FiLM', 'Condition17', 'train_dataset_13', 'val_dataset_11', 'selectable_preprocessed_contract', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260820_220340`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=798，val=505，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [13, 11]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，input_size=`[90, 56]` (W,H)，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,854,147 |
| 训练 | epochs=50，batch=64，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 41） | 10.504 | 33.284 mm | 29.828 mm | 19.357 mm | 23.003 mm |
| 最终训练（epoch 50） | 5.454 | 22.897 mm | 20.535 mm | 13.966 mm | 15.147 mm |
| 最终验证（epoch 50） | 10.612 | 33.369 mm | 29.898 mm | 19.063 mm | 23.578 mm |

相比上一实验：最佳 Val EPE 下降 0.120 mm，性能提升。

## 2026-08-20T23:24:17+08:00 · 20260820_230711

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260820_230711","completed_at":"2026-08-20T23:24:17+08:00","run_name":"20260820_230711","best_val_epe_mm":32.18024826049805,"best_epoch":42} -->

- 实验目的：Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy eye images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from Condition17=[TableFrame7_N, log virtual-distance scale, virtual camera pose in table frame].
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'direct_table_UV', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'virtual_pose_FiLM', 'Condition17', 'train_dataset_13', 'val_dataset_11', 'selectable_preprocessed_contract', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260820_230711`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=798，val=505，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [13, 11]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，input_size=`[90, 56]` (W,H)，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=36,247,746 |
| 训练 | epochs=50，batch=64，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 42） | 9.843 | 32.180 mm | 28.770 mm | 19.148 mm | 21.821 mm |
| 最终训练（epoch 50） | 5.098 | 22.110 mm | 20.787 mm | 13.948 mm | 14.222 mm |
| 最终验证（epoch 50） | 10.012 | 32.338 mm | 28.717 mm | 19.258 mm | 22.019 mm |

相比上一实验：最佳 Val EPE 下降 1.104 mm，性能提升。

## 2026-08-21T00:08:45+08:00 · 20260820_235025

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260820_235025","completed_at":"2026-08-21T00:08:45+08:00","run_name":"20260820_235025","best_val_epe_mm":33.62167739868164,"best_epoch":16} -->

- 实验目的：Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy eye images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from Condition17=[TableFrame7_N, log virtual-distance scale, virtual camera pose in table frame].
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'direct_table_UV', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'virtual_pose_FiLM', 'Condition17', 'train_dataset_13', 'val_dataset_11', 'selectable_preprocessed_contract', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260820_235025`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=798，val=505，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [13, 11]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，input_size=`[90, 56]` (W,H)，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=36,314,050 |
| 训练 | epochs=50，batch=64，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 16） | 10.521 | 33.622 mm | 30.417 mm | 20.546 mm | 22.309 mm |
| 最终训练（epoch 50） | 4.929 | 21.719 mm | 20.250 mm | 13.470 mm | 14.147 mm |
| 最终验证（epoch 50） | 11.435 | 34.873 mm | 31.353 mm | 22.507 mm | 22.025 mm |

相比上一实验：最佳 Val EPE 上升 1.441 mm，性能下降。

## 2026-08-21T11:35:21+08:00 · 20260821_111758

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260821_111758","completed_at":"2026-08-21T11:35:21+08:00","run_name":"20260821_111758","best_val_epe_mm":32.15056228637695,"best_epoch":50} -->

- 实验目的：Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy eye images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from Condition17=[TableFrame7_N, log virtual-distance scale, virtual camera pose in table frame].
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'direct_table_UV', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'virtual_pose_FiLM', 'Condition17', 'train_dataset_13', 'val_dataset_11', 'selectable_preprocessed_contract', 'no_DECA', 'H5_small_stem_fpn24', 'H7_iris_heatmap_auxiliary']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260821_111758`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=798，val=505，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [13, 11]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，input_size=`[90, 56]` (W,H)，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=36,543,043 |
| 训练 | epochs=50，batch=64，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 50） | 9.923 | 32.151 mm | 28.906 mm | 19.274 mm | 21.541 mm |
| 最终训练（epoch 50） | 5.197 | 22.124 mm | 19.664 mm | 13.513 mm | 14.581 mm |
| 最终验证（epoch 50） | 9.923 | 32.151 mm | 28.906 mm | 19.274 mm | 21.541 mm |

相比上一实验：最佳 Val EPE 下降 1.471 mm，性能提升。

## 2026-08-21T15:33:48+08:00 · 20260821_151148

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260821_151148","completed_at":"2026-08-21T15:33:48+08:00","run_name":"20260821_151148","best_val_epe_mm":32.15056228637695,"best_epoch":50} -->

- 实验目的：Preprocessed-v2 direct table-UV experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy eye images; directly output normalized table-local (u,v), use only inverse target normalization before the millimetre Smooth L1/EPE, encode TableFrame7 in N, and condition fusion with FiLM from Condition17=[TableFrame7_N, log virtual-distance scale, virtual camera pose in table frame].
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'direct_table_UV', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'virtual_pose_FiLM', 'Condition17', 'train_dataset_13', 'val_dataset_11', 'selectable_preprocessed_contract', 'no_DECA', 'H5_small_stem_fpn24', 'H7_iris_heatmap_auxiliary']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260821_151148`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=798，val=505，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [13, 11]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，input_size=`[90, 56]` (W,H)，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=36,543,043 |
| 训练 | epochs=50，batch=64，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 50） | 9.923 | 32.151 mm | 28.906 mm | 19.274 mm | 21.541 mm |
| 最终训练（epoch 50） | 5.197 | 22.124 mm | 19.664 mm | 13.513 mm | 14.581 mm |
| 最终验证（epoch 50） | 9.923 | 32.151 mm | 28.906 mm | 19.274 mm | 21.541 mm |

相比上一实验：最佳 Val EPE 与上一实验相同。

## 2026-08-21T16:23:36+08:00 · 20260821_160335

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260821_160335","completed_at":"2026-08-21T16:23:36+08:00","run_name":"20260821_160335","best_val_epe_mm":33.163795471191406,"best_epoch":45} -->

- 实验目的：Preprocessed-v2 direct table-UV H5+F1 experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy 90x56 eye images; Eye uses H5 small stem + multiscale FPN 4x6 tokens, Face uses F1 padded mixed_7a 4x4 tokens, then direct table-local (u,v) is predicted with virtual-distance and virtual-pose FiLM.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'direct_table_UV', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'virtual_pose_FiLM', 'Condition17', 'train_dataset_13', 'val_dataset_11', 'selectable_preprocessed_contract', 'no_DECA', 'H5_small_stem_fpn24', 'F1_padded_mixed7a_4x4', 'face_tokens_4x4', 'no_H6_binocular_self_attention', 'no_H7_iris_heatmap_auxiliary']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260821_160335`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=798，val=505，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [13, 11]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，input_size=`[90, 56]` (W,H)，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=36,248,642 |
| 训练 | epochs=50，batch=64，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 45） | 10.414 | 33.164 mm | 30.294 mm | 20.623 mm | 21.335 mm |
| 最终训练（epoch 50） | 5.051 | 22.047 mm | 20.674 mm | 13.527 mm | 14.519 mm |
| 最终验证（epoch 50） | 10.461 | 33.193 mm | 29.459 mm | 20.521 mm | 21.633 mm |

相比上一实验：最佳 Val EPE 上升 1.013 mm，性能下降。

## 2026-08-21T18:25:25+08:00 · 20260821_180729

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260821_180729","completed_at":"2026-08-21T18:25:25+08:00","run_name":"20260821_180729","best_val_epe_mm":32.18024826049805,"best_epoch":42} -->

- 实验目的：Preprocessed-v2 direct table-UV H5+F0 experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy 90x56 eye images; Eye uses H5 small stem + multiscale FPN 4x6 tokens, Face uses the native Inception-ResNet V1 3x3 spatial tokens, then direct table-local (u,v) is predicted with virtual-distance and virtual-pose FiLM.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'direct_table_UV', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'virtual_pose_FiLM', 'Condition17', 'train_dataset_13', 'val_dataset_11', 'selectable_preprocessed_contract', 'no_DECA', 'H5_small_stem_fpn24', 'F0_native_3x3', 'face_tokens_3x3', 'no_H6_binocular_self_attention', 'no_H7_iris_heatmap_auxiliary']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260821_180729`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=798，val=505，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [13, 11]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，input_size=`[90, 56]` (W,H)，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=36,247,746 |
| 训练 | epochs=50，batch=64，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 42） | 9.843 | 32.180 mm | 28.770 mm | 19.148 mm | 21.821 mm |
| 最终训练（epoch 50） | 5.098 | 22.110 mm | 20.787 mm | 13.948 mm | 14.222 mm |
| 最终验证（epoch 50） | 10.012 | 32.338 mm | 28.717 mm | 19.258 mm | 22.019 mm |

相比上一实验：最佳 Val EPE 下降 0.984 mm，性能提升。

## 2026-08-23T21:52:25+08:00 · 20260823_212547

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260823_212547","completed_at":"2026-08-23T21:52:25+08:00","run_name":"20260823_212547","best_val_epe_mm":32.0848388671875,"best_epoch":21} -->

- 实验目的：Preprocessed-v2 direct table-UV H5+F0 experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy 90x56 eye images; Eye uses H5 small stem + multiscale FPN 4x6 tokens, Face uses the native Inception-ResNet V1 3x3 spatial tokens, then direct table-local (u,v) is predicted with virtual-distance and virtual-pose FiLM.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'direct_table_UV', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'virtual_pose_FiLM', 'Condition17', 'train_dataset_13', 'val_dataset_11', 'selectable_preprocessed_contract', 'no_DECA', 'H5_small_stem_fpn24', 'F0_native_3x3', 'face_tokens_3x3', 'no_H6_binocular_self_attention', 'probabilistic_15_point_eye_keypoints', 'full_2d_keypoint_covariance']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260823_212547`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=798，val=505，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [13, 11]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，input_size=`[90, 56]` (W,H)，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=36,752,222 |
| 训练 | epochs=50，batch=64，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 21） | 9.877 | 32.085 mm | 30.333 mm | 19.832 mm | 20.965 mm |
| 最终训练（epoch 50） | 5.368 | 22.115 mm | 20.317 mm | 13.480 mm | 14.568 mm |
| 最终验证（epoch 50） | 10.128 | 32.177 mm | 28.301 mm | 19.545 mm | 21.334 mm |

相比上一实验：最佳 Val EPE 下降 0.095 mm，性能提升。

## 2026-08-24T10:02:06+08:00 · 20260824_093535

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260824_093535","completed_at":"2026-08-24T10:02:06+08:00","run_name":"20260824_093535","best_val_epe_mm":32.24542999267578,"best_epoch":37} -->

- 实验目的：Preprocessed-v2 direct table-UV H5+F0 experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy 90x56 eye images; Eye uses H5 small stem + multiscale FPN 4x6 tokens, Face uses the native Inception-ResNet V1 3x3 spatial tokens, then direct table-local (u,v) is predicted with virtual-distance and virtual-pose FiLM.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'direct_table_UV', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'virtual_pose_FiLM', 'Condition17', 'train_dataset_13', 'val_dataset_11', 'selectable_preprocessed_contract', 'no_DECA', 'H5_small_stem_fpn24', 'F0_native_3x3', 'face_tokens_3x3', 'no_H6_binocular_self_attention', 'probabilistic_15_point_eye_keypoints', 'full_2d_keypoint_covariance']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260824_093535`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=798，val=505，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [13, 11]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，input_size=`[90, 56]` (W,H)，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=36,753,381 |
| 训练 | epochs=50，batch=64，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 37） | 10.124 | 32.245 mm | 28.928 mm | 19.563 mm | 21.682 mm |
| 最终训练（epoch 50） | 5.201 | 21.789 mm | 19.909 mm | 12.800 mm | 15.068 mm |
| 最终验证（epoch 50） | 10.142 | 32.334 mm | 28.774 mm | 19.610 mm | 21.606 mm |

相比上一实验：最佳 Val EPE 上升 0.161 mm，性能下降。

## 2026-08-24T20:59:08+08:00 · 20260824_203558

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260824_203558","completed_at":"2026-08-24T20:59:08+08:00","run_name":"20260824_203558","best_val_epe_mm":32.12826919555664,"best_epoch":37} -->

- 实验目的：Preprocessed-v2 direct table-UV H5+F0 experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy 90x56 eye images; Eye uses H5 small stem + multiscale FPN 4x6 tokens, Face uses the native Inception-ResNet V1 3x3 spatial tokens, then direct table-local (u,v) is predicted with virtual-distance and virtual-pose FiLM.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'direct_table_UV', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'virtual_pose_FiLM', 'Condition17', 'train_dataset_13', 'val_dataset_11', 'selectable_preprocessed_contract', 'no_DECA', 'H5_small_stem_fpn24', 'F0_native_3x3', 'face_tokens_3x3', 'no_H6_binocular_self_attention', 'probabilistic_15_point_eye_keypoints', 'full_2d_keypoint_covariance']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260824_203558`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=798，val=505，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [13, 11]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，input_size=`[90, 56]` (W,H)，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=36,753,381 |
| 训练 | epochs=50，batch=64，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 37） | 10.070 | 32.128 mm | 28.703 mm | 19.523 mm | 21.551 mm |
| 最终训练（epoch 50） | 5.220 | 21.828 mm | 19.947 mm | 12.793 mm | 15.109 mm |
| 最终验证（epoch 50） | 10.099 | 32.208 mm | 28.420 mm | 19.536 mm | 21.447 mm |

相比上一实验：最佳 Val EPE 下降 0.117 mm，性能提升。

## 2026-08-25T13:24:05+08:00 · 20260825_130035

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260825_130035","completed_at":"2026-08-25T13:24:05+08:00","run_name":"20260825_130035","best_val_epe_mm":32.577667236328125,"best_epoch":42} -->

- 实验目的：Preprocessed-v2 direct table-UV H5+F0 experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy 90x56 eye images; Eye uses H5 small stem + multiscale FPN 4x6 tokens, Face uses the native Inception-ResNet V1 3x3 spatial tokens, then direct table-local (u,v) is predicted with virtual-distance and virtual-pose FiLM.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'direct_table_UV', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'virtual_pose_FiLM', 'Condition17', 'train_dataset_13', 'val_dataset_11', 'selectable_preprocessed_contract', 'no_DECA', 'H5_small_stem_fpn24', 'F0_native_3x3', 'face_tokens_3x3', 'no_H6_binocular_self_attention', 'probabilistic_15_point_eye_keypoints', 'full_2d_keypoint_covariance']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260825_130035`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=798，val=505，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [13, 11]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，input_size=`[90, 56]` (W,H)，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=36,986,045 |
| 训练 | epochs=50，batch=64，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 42） | 12.966 | 32.578 mm | 29.201 mm | 20.224 mm | 21.334 mm |
| 最终训练（epoch 50） | 4.792 | 18.686 mm | 16.807 mm | 12.951 mm | 10.813 mm |
| 最终验证（epoch 50） | 13.742 | 33.616 mm | 29.610 mm | 20.464 mm | 22.371 mm |

相比上一实验：最佳 Val EPE 上升 0.449 mm，性能下降。

## 2026-08-25T14:40:16+08:00 · 20260825_140932

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260825_140932","completed_at":"2026-08-25T14:40:16+08:00","run_name":"20260825_140932","best_val_epe_mm":31.11603546142578,"best_epoch":43} -->

- 实验目的：Preprocessed-v2 direct table-UV H5+F0 experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy 90x56 eye images; Eye uses H5 small stem + multiscale FPN 4x6 tokens, Face uses the native Inception-ResNet V1 3x3 spatial tokens, then direct table-local (u,v) is predicted with virtual-distance and virtual-pose FiLM.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'direct_table_UV', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'virtual_pose_FiLM', 'Condition17', 'train_dataset_13', 'val_dataset_11', 'selectable_preprocessed_contract', 'no_DECA', 'H5_small_stem_fpn24', 'F0_native_3x3', 'face_tokens_3x3', 'no_H6_binocular_self_attention', 'probabilistic_15_point_eye_keypoints', 'full_2d_keypoint_covariance']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260825_140932`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=798，val=505，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [13, 11]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，input_size=`[90, 56]` (W,H)，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=36,967,325 |
| 训练 | epochs=50，batch=64，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 43） | 11.954 | 31.116 mm | 28.095 mm | 19.165 mm | 20.288 mm |
| 最终训练（epoch 50） | 5.535 | 20.737 mm | 19.020 mm | 15.366 mm | 10.875 mm |
| 最终验证（epoch 50） | 12.070 | 31.305 mm | 27.949 mm | 19.144 mm | 20.525 mm |

相比上一实验：最佳 Val EPE 下降 1.462 mm，性能提升。

## 2026-08-25T22:39:18+08:00 · 20260825_211030

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260825_211030","completed_at":"2026-08-25T22:39:18+08:00","run_name":"20260825_211030","best_val_epe_mm":32.02960205078125,"best_epoch":39} -->

- 实验目的：Preprocessed-v2 direct table-UV H5+F0 experiment: train Dataset 13 and validate Dataset 11 with virtual-camera face images and legacy 90x56 eye images; Eye uses H5 small stem + multiscale FPN 4x6 tokens, Face uses the native Inception-ResNet V1 3x3 spatial tokens, then direct table-local (u,v) is predicted with virtual-distance and virtual-pose FiLM.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'direct_uv', 'direct_table_UV', 'preprocessed_v2', 'virtual_camera', 'TableFrame7_N', 'virtual_distance_FiLM', 'virtual_pose_FiLM', 'Condition17', 'train_dataset_13', 'val_dataset_11', 'selectable_preprocessed_contract', 'no_DECA', 'H5_small_stem_fpn24', 'F0_native_3x3', 'face_tokens_3x3', 'no_H6_binocular_self_attention', 'probabilistic_15_point_eye_keypoints', 'full_2d_keypoint_covariance']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260825_211030`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=798，val=505，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [13, 11]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，input_size=`[90, 56]` (W,H)，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=36,572,829 |
| 训练 | epochs=50，batch=64，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.0 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 39） | 12.842 | 32.030 mm | 28.938 mm | 18.720 mm | 21.987 mm |
| 最终训练（epoch 50） | 5.590 | 20.598 mm | 19.042 mm | 14.347 mm | 11.948 mm |
| 最终验证（epoch 50） | 13.038 | 32.621 mm | 30.029 mm | 19.262 mm | 21.886 mm |

相比上一实验：最佳 Val EPE 上升 0.914 mm，性能下降。

<!-- AUTO-DETAILS-END -->
