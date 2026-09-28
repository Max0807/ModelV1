# 新旧预处理前 5 样本字段对照：dataset11

本报告按你的要求执行：以新 CSV 字段为准，旧 CSV 的冗余字段不检查；每个可对齐字段只展示前 5 个样本的具体值，不计算最大误差或平均误差。

## 文件与行数

- 新 Stage 1 base: 583 rows, `data/preprocessed_table7/dataset11/stage1_base/base_dataset11.csv`
- 旧 shared modelv1: 583 rows, `data/preprocessed_v2_160/dataset11/shared/modelv1_dataset11.csv`
- 新 Stage 2B metric pose: 583 rows, `data/preprocessed_table7/dataset11/stage2_metric_head_pose/iris65_metric_head_pose_dataset11.csv`
- 旧 Iris65 depth priors: 583 rows, `data/preprocessed_v2_160/dataset11/iris65/depth_priors_iris65_pnp_v2_dataset11.csv`
- 新 Stage 3 final training: 578 rows, `data/preprocessed_table7/dataset11/stage3_virtual_table7/training_table7_dataset11.csv`
- 旧 common modelv1: 505 rows, `data/preprocessed_v2_160/dataset11/comparison/modelv1_dataset11_common_pnp1010_iris65.csv`
- 旧 virtual camera manifest: 505 rows, `data/preprocessed_v2_160/dataset11/virtual_camera/common_pnp1010_iris65/manifest.csv`

## JSON 摘要

- 新 Stage 1 report: schema_version=modelv1_table7_base_v2; dataset=dataset_dual_rigid_body_11; input_rows=583; written_rows=583; table_z_source=per-sample-target
- 旧 shared report: table_z_source=per-sample-target
- 新 Stage 2B report: schema_version=iris65_metric_head_pose_v2_rotation_only; counts={'success': 578, 'failed': 5}
- 新 Stage 3 report: schema_version=modelv1_virtual_table7_training_v1; counts={'input': 583, 'success': 578, 'failed': 5}; failure_types={'ValueError': 5}; virtual_camera_center_source=Iris65-derived metric head origin
- 旧 manifest report: (no selected summary keys)

## Stage 1 base CSV vs 旧 shared/modelv1 CSV

- 新 CSV 行数：583
- 旧对照 CSV 行数：583
- 本节抽样：前 5 个在旧文件中也存在的样本
- 样本：`dataset_dual_rigid_body_11/img_20260805_114114_00000`, `dataset_dual_rigid_body_11/img_20260805_114116_00001`, `dataset_dual_rigid_body_11/img_20260805_114122_00004`, `dataset_dual_rigid_body_11/img_20260805_114125_00008`, `dataset_dual_rigid_body_11/img_20260805_114127_00009`
- 抽样字段判断正常：59 个；出现 different：0 个；旧值缺失：0 个。
- 本节覆盖新 Stage 1 CSV 的全部字段；旧文件没有同名的 R/table 字段时，从旧 scene_00..24 反推。
- face_path 的目录名不同：新配置使用 insightface_face，旧流程使用 insightface_face_160；文件名一致。

| 新字段 | 旧字段/来源 | 样本 | 新值 | 旧值 | 判断 |
|---|---|---|---|---|---|
| `sample_id` | `sample_id` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | same |
| `sample_id` | `sample_id` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | same |
| `sample_id` | `sample_id` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | same |
| `sample_id` | `sample_id` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | same |
| `sample_id` | `sample_id` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | same |
| `dataset` | `dataset` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `dataset_dual_rigid_body_11` | `dataset_dual_rigid_body_11` | same |
| `dataset` | `dataset` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `dataset_dual_rigid_body_11` | `dataset_dual_rigid_body_11` | same |
| `dataset` | `dataset` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `dataset_dual_rigid_body_11` | `dataset_dual_rigid_body_11` | same |
| `dataset` | `dataset` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `dataset_dual_rigid_body_11` | `dataset_dual_rigid_body_11` | same |
| `dataset` | `dataset` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `dataset_dual_rigid_body_11` | `dataset_dual_rigid_body_11` | same |
| `image_name` | `image_name` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `img_20260805_114114_00000.jpg` | `img_20260805_114114_00000.jpg` | same |
| `image_name` | `image_name` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `img_20260805_114116_00001.jpg` | `img_20260805_114116_00001.jpg` | same |
| `image_name` | `image_name` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `img_20260805_114122_00004.jpg` | `img_20260805_114122_00004.jpg` | same |
| `image_name` | `image_name` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `img_20260805_114125_00008.jpg` | `img_20260805_114125_00008.jpg` | same |
| `image_name` | `image_name` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `img_20260805_114127_00009.jpg` | `img_20260805_114127_00009.jpg` | same |
| `frame_idx` | `frame_idx` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `0` | `0` | same |
| `frame_idx` | `frame_idx` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `1` | `1` | same |
| `frame_idx` | `frame_idx` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `4` | `4` | same |
| `frame_idx` | `frame_idx` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `8` | `8` | same |
| `frame_idx` | `frame_idx` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `9` | `9` | same |
| `source_dataset_dir` | `source_dataset_dir` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `D:\GithubCode\CrossGaze-main\CrossGaz ... collection\dataset_dual_rigid_body_11` | `D:\GithubCode\CrossGaze-main\CrossGaz ... collection\dataset_dual_rigid_body_11` | same |
| `source_dataset_dir` | `source_dataset_dir` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `D:\GithubCode\CrossGaze-main\CrossGaz ... collection\dataset_dual_rigid_body_11` | `D:\GithubCode\CrossGaze-main\CrossGaz ... collection\dataset_dual_rigid_body_11` | same |
| `source_dataset_dir` | `source_dataset_dir` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `D:\GithubCode\CrossGaze-main\CrossGaz ... collection\dataset_dual_rigid_body_11` | `D:\GithubCode\CrossGaze-main\CrossGaz ... collection\dataset_dual_rigid_body_11` | same |
| `source_dataset_dir` | `source_dataset_dir` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `D:\GithubCode\CrossGaze-main\CrossGaz ... collection\dataset_dual_rigid_body_11` | `D:\GithubCode\CrossGaze-main\CrossGaz ... collection\dataset_dual_rigid_body_11` | same |
| `source_dataset_dir` | `source_dataset_dir` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `D:\GithubCode\CrossGaze-main\CrossGaz ... collection\dataset_dual_rigid_body_11` | `D:\GithubCode\CrossGaze-main\CrossGaz ... collection\dataset_dual_rigid_body_11` | same |
| `source_image_path` | `source_image_path` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `D:\GithubCode\CrossGaze-main\CrossGaz ... ace_img\img_20260805_114114_00000.jpg` | `D:\GithubCode\CrossGaze-main\CrossGaz ... ace_img\img_20260805_114114_00000.jpg` | same |
| `source_image_path` | `source_image_path` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `D:\GithubCode\CrossGaze-main\CrossGaz ... ace_img\img_20260805_114116_00001.jpg` | `D:\GithubCode\CrossGaze-main\CrossGaz ... ace_img\img_20260805_114116_00001.jpg` | same |
| `source_image_path` | `source_image_path` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `D:\GithubCode\CrossGaze-main\CrossGaz ... ace_img\img_20260805_114122_00004.jpg` | `D:\GithubCode\CrossGaze-main\CrossGaz ... ace_img\img_20260805_114122_00004.jpg` | same |
| `source_image_path` | `source_image_path` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `D:\GithubCode\CrossGaze-main\CrossGaz ... ace_img\img_20260805_114125_00008.jpg` | `D:\GithubCode\CrossGaze-main\CrossGaz ... ace_img\img_20260805_114125_00008.jpg` | same |
| `source_image_path` | `source_image_path` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `D:\GithubCode\CrossGaze-main\CrossGaz ... ace_img\img_20260805_114127_00009.jpg` | `D:\GithubCode\CrossGaze-main\CrossGaz ... ace_img\img_20260805_114127_00009.jpg` | same |
| `face_path` | `face_path` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `D:\GithubCode\CrossGaze-main\CrossGaz ... ce_face\img_20260805_114114_00000.jpg` | `D:\GithubCode\CrossGaze-main\CrossGaz ... ace_160\img_20260805_114114_00000.jpg` | basename_same |
| `face_path` | `face_path` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `D:\GithubCode\CrossGaze-main\CrossGaz ... ce_face\img_20260805_114116_00001.jpg` | `D:\GithubCode\CrossGaze-main\CrossGaz ... ace_160\img_20260805_114116_00001.jpg` | basename_same |
| `face_path` | `face_path` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `D:\GithubCode\CrossGaze-main\CrossGaz ... ce_face\img_20260805_114122_00004.jpg` | `D:\GithubCode\CrossGaze-main\CrossGaz ... ace_160\img_20260805_114122_00004.jpg` | basename_same |
| `face_path` | `face_path` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `D:\GithubCode\CrossGaze-main\CrossGaz ... ce_face\img_20260805_114125_00008.jpg` | `D:\GithubCode\CrossGaze-main\CrossGaz ... ace_160\img_20260805_114125_00008.jpg` | basename_same |
| `face_path` | `face_path` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `D:\GithubCode\CrossGaze-main\CrossGaz ... ce_face\img_20260805_114127_00009.jpg` | `D:\GithubCode\CrossGaze-main\CrossGaz ... ace_160\img_20260805_114127_00009.jpg` | basename_same |
| `face_bbox_x` | `face_bbox_x` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `937` | `937` | same |
| `face_bbox_x` | `face_bbox_x` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `936` | `936` | same |
| `face_bbox_x` | `face_bbox_x` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `934` | `934` | same |
| `face_bbox_x` | `face_bbox_x` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `934` | `934` | same |
| `face_bbox_x` | `face_bbox_x` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `943` | `943` | same |
| `face_bbox_y` | `face_bbox_y` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `193` | `193` | same |
| `face_bbox_y` | `face_bbox_y` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `198` | `198` | same |
| `face_bbox_y` | `face_bbox_y` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `202` | `202` | same |
| `face_bbox_y` | `face_bbox_y` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `208` | `208` | same |
| `face_bbox_y` | `face_bbox_y` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `214` | `214` | same |
| `face_bbox_w` | `face_bbox_w` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `311` | `311` | same |
| `face_bbox_w` | `face_bbox_w` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `311` | `311` | same |
| `face_bbox_w` | `face_bbox_w` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `324` | `324` | same |
| `face_bbox_w` | `face_bbox_w` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `325` | `325` | same |
| `face_bbox_w` | `face_bbox_w` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `322` | `322` | same |
| `face_bbox_h` | `face_bbox_h` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `434` | `434` | same |
| `face_bbox_h` | `face_bbox_h` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `428` | `428` | same |
| `face_bbox_h` | `face_bbox_h` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `442` | `442` | same |
| `face_bbox_h` | `face_bbox_h` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `438` | `438` | same |
| `face_bbox_h` | `face_bbox_h` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `430` | `430` | same |
| `left_eye_bbox_x` | `left_eye_bbox_x` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `1169` | `1169` | same |
| `left_eye_bbox_x` | `left_eye_bbox_x` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `1169` | `1169` | same |
| `left_eye_bbox_x` | `left_eye_bbox_x` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `1172` | `1172` | same |
| `left_eye_bbox_x` | `left_eye_bbox_x` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `1172` | `1172` | same |
| `left_eye_bbox_x` | `left_eye_bbox_x` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `1182` | `1182` | same |
| `left_eye_bbox_y` | `left_eye_bbox_y` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `348` | `348` | same |
| `left_eye_bbox_y` | `left_eye_bbox_y` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `348` | `348` | same |
| `left_eye_bbox_y` | `left_eye_bbox_y` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `363` | `363` | same |
| `left_eye_bbox_y` | `left_eye_bbox_y` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `368` | `368` | same |
| `left_eye_bbox_y` | `left_eye_bbox_y` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `373` | `373` | same |
| `left_eye_bbox_w` | `left_eye_bbox_w` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `90` | `90` | same |
| `left_eye_bbox_w` | `left_eye_bbox_w` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `90` | `90` | same |
| `left_eye_bbox_w` | `left_eye_bbox_w` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `90` | `90` | same |
| `left_eye_bbox_w` | `left_eye_bbox_w` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `90` | `90` | same |
| `left_eye_bbox_w` | `left_eye_bbox_w` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `90` | `90` | same |
| `left_eye_bbox_h` | `left_eye_bbox_h` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `54` | `54` | same |
| `left_eye_bbox_h` | `left_eye_bbox_h` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `54` | `54` | same |
| `left_eye_bbox_h` | `left_eye_bbox_h` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `54` | `54` | same |
| `left_eye_bbox_h` | `left_eye_bbox_h` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `54` | `54` | same |
| `left_eye_bbox_h` | `left_eye_bbox_h` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `54` | `54` | same |
| `right_eye_bbox_x` | `right_eye_bbox_x` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `1033` | `1033` | same |
| `right_eye_bbox_x` | `right_eye_bbox_x` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `1032` | `1032` | same |
| `right_eye_bbox_x` | `right_eye_bbox_x` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `1032` | `1032` | same |
| `right_eye_bbox_x` | `right_eye_bbox_x` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `1031` | `1031` | same |
| `right_eye_bbox_x` | `right_eye_bbox_x` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `1045` | `1045` | same |
| `right_eye_bbox_y` | `right_eye_bbox_y` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `330` | `330` | same |
| `right_eye_bbox_y` | `right_eye_bbox_y` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `331` | `331` | same |
| `right_eye_bbox_y` | `right_eye_bbox_y` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `339` | `339` | same |
| `right_eye_bbox_y` | `right_eye_bbox_y` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `340` | `340` | same |
| `right_eye_bbox_y` | `right_eye_bbox_y` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `342` | `342` | same |
| `right_eye_bbox_w` | `right_eye_bbox_w` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `90` | `90` | same |
| `right_eye_bbox_w` | `right_eye_bbox_w` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `90` | `90` | same |
| `right_eye_bbox_w` | `right_eye_bbox_w` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `90` | `90` | same |
| `right_eye_bbox_w` | `right_eye_bbox_w` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `90` | `90` | same |
| `right_eye_bbox_w` | `right_eye_bbox_w` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `90` | `90` | same |
| `right_eye_bbox_h` | `right_eye_bbox_h` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `54` | `54` | same |
| `right_eye_bbox_h` | `right_eye_bbox_h` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `54` | `54` | same |
| `right_eye_bbox_h` | `right_eye_bbox_h` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `54` | `54` | same |
| `right_eye_bbox_h` | `right_eye_bbox_h` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `54` | `54` | same |
| `right_eye_bbox_h` | `right_eye_bbox_h` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `54` | `54` | same |
| `z_table_mm` | `z_table_mm` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `15.2317` | `15.2317` | same |
| `z_table_mm` | `z_table_mm` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `15.1546` | `15.1546` | same |
| `z_table_mm` | `z_table_mm` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `15.0524` | `15.0524` | same |
| `z_table_mm` | `z_table_mm` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `15.0531` | `15.0531` | same |
| `z_table_mm` | `z_table_mm` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `15.1732` | `15.1732` | same |
| `uv_gt_u_mm` | `uv_gt_u_mm` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `-82.33254255` | `-82.33254255` | same |
| `uv_gt_u_mm` | `uv_gt_u_mm` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `-91.94431042` | `-91.94431042` | same |
| `uv_gt_u_mm` | `uv_gt_u_mm` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `-92.0335085` | `-92.0335085` | same |
| `uv_gt_u_mm` | `uv_gt_u_mm` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `-92.13032686` | `-92.13032686` | same |
| `uv_gt_u_mm` | `uv_gt_u_mm` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `-110.9464735` | `-110.9464735` | same |
| `uv_gt_v_mm` | `uv_gt_v_mm` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `263.1358178` | `263.1358178` | same |
| `uv_gt_v_mm` | `uv_gt_v_mm` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `263.0778118` | `263.0778118` | same |
| `uv_gt_v_mm` | `uv_gt_v_mm` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `262.8666208` | `262.8666208` | same |
| `uv_gt_v_mm` | `uv_gt_v_mm` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `262.7871002` | `262.7871002` | same |
| `uv_gt_v_mm` | `uv_gt_v_mm` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `264.0870756` | `264.0870756` | same |
| `gaze_target_w_x_mm` | `gaze_target_w_x_mm` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `230.0296` | `230.0296` | same |
| `gaze_target_w_x_mm` | `gaze_target_w_x_mm` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `220.3771` | `220.3771` | same |
| `gaze_target_w_x_mm` | `gaze_target_w_x_mm` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `220.3511` | `220.3511` | same |
| `gaze_target_w_x_mm` | `gaze_target_w_x_mm` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `220.3665` | `220.3665` | same |
| `gaze_target_w_x_mm` | `gaze_target_w_x_mm` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `201.5291` | `201.5291` | same |
| `gaze_target_w_y_mm` | `gaze_target_w_y_mm` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `243.9548` | `243.9548` | same |
| `gaze_target_w_y_mm` | `gaze_target_w_y_mm` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `243.8841` | `243.8841` | same |
| `gaze_target_w_y_mm` | `gaze_target_w_y_mm` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `243.74` | `243.74` | same |
| `gaze_target_w_y_mm` | `gaze_target_w_y_mm` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `243.7625` | `243.7625` | same |
| `gaze_target_w_y_mm` | `gaze_target_w_y_mm` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `244.9458` | `244.9458` | same |
| `gaze_target_w_z_mm` | `gaze_target_w_z_mm` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `15.2317` | `15.2317` | same |
| `gaze_target_w_z_mm` | `gaze_target_w_z_mm` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `15.1546` | `15.1546` | same |
| `gaze_target_w_z_mm` | `gaze_target_w_z_mm` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `15.0524` | `15.0524` | same |
| `gaze_target_w_z_mm` | `gaze_target_w_z_mm` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `15.0531` | `15.0531` | same |
| `gaze_target_w_z_mm` | `gaze_target_w_z_mm` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `15.1732` | `15.1732` | same |
| `gaze_cam_recomputed_x_mm` | `gaze_cam_recomputed_x_mm` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `257.7816584` | `257.7816584` | same |
| `gaze_cam_recomputed_x_mm` | `gaze_cam_recomputed_x_mm` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `257.2413446` | `257.2413446` | same |
| `gaze_cam_recomputed_x_mm` | `gaze_cam_recomputed_x_mm` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `256.9567913` | `256.9567913` | same |
| `gaze_cam_recomputed_x_mm` | `gaze_cam_recomputed_x_mm` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `256.7405196` | `256.7405196` | same |
| `gaze_cam_recomputed_x_mm` | `gaze_cam_recomputed_x_mm` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `257.2133571` | `257.2133571` | same |
| `gaze_cam_recomputed_y_mm` | `gaze_cam_recomputed_y_mm` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `72.65092867` | `72.65092867` | same |
| `gaze_cam_recomputed_y_mm` | `gaze_cam_recomputed_y_mm` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `76.4044196` | `76.4044196` | same |
| `gaze_cam_recomputed_y_mm` | `gaze_cam_recomputed_y_mm` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `76.56886405` | `76.56886405` | same |
| `gaze_cam_recomputed_y_mm` | `gaze_cam_recomputed_y_mm` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `76.10464969` | `76.10464969` | same |
| `gaze_cam_recomputed_y_mm` | `gaze_cam_recomputed_y_mm` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `83.30219686` | `83.30219686` | same |
| `gaze_cam_recomputed_z_mm` | `gaze_cam_recomputed_z_mm` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `72.1278127` | `72.1278127` | same |
| `gaze_cam_recomputed_z_mm` | `gaze_cam_recomputed_z_mm` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `81.0617642` | `81.0617642` | same |
| `gaze_cam_recomputed_z_mm` | `gaze_cam_recomputed_z_mm` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `81.24588766` | `81.24588766` | same |
| `gaze_cam_recomputed_z_mm` | `gaze_cam_recomputed_z_mm` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `82.09626118` | `82.09626118` | same |
| `gaze_cam_recomputed_z_mm` | `gaze_cam_recomputed_z_mm` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `99.26195114` | `99.26195114` | same |
| `t_wc_x_mm` | `t_wc_x_mm` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `312.3621426` | `312.3621426` | same |
| `t_wc_x_mm` | `t_wc_x_mm` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `312.3214104` | `312.3214104` | same |
| `t_wc_x_mm` | `t_wc_x_mm` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `312.3846085` | `312.3846085` | same |
| `t_wc_x_mm` | `t_wc_x_mm` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `312.4968269` | `312.4968269` | same |
| `t_wc_x_mm` | `t_wc_x_mm` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `312.4755735` | `312.4755735` | same |
| `t_wc_y_mm` | `t_wc_y_mm` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `-19.1810178` | `-19.1810178` | same |
| `t_wc_y_mm` | `t_wc_y_mm` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `-19.19371184` | `-19.19371184` | same |
| `t_wc_y_mm` | `t_wc_y_mm` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `-19.12662083` | `-19.12662083` | same |
| `t_wc_y_mm` | `t_wc_y_mm` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `-19.02460021` | `-19.02460021` | same |
| `t_wc_y_mm` | `t_wc_y_mm` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `-19.1412756` | `-19.1412756` | same |
| `t_wc_z_mm` | `t_wc_z_mm` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `45.44490522` | `45.44490522` | same |
| `t_wc_z_mm` | `t_wc_z_mm` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `45.45448097` | `45.45448097` | same |
| `t_wc_z_mm` | `t_wc_z_mm` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `45.40841941` | `45.40841941` | same |
| `t_wc_z_mm` | `t_wc_z_mm` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `45.09252585` | `45.09252585` | same |
| `t_wc_z_mm` | `t_wc_z_mm` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `45.16857458` | `45.16857458` | same |
| `t_cw_x_mm` | `t_cw_x_mm` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `2.432960451` | `2.432960451` | same |
| `t_cw_x_mm` | `t_cw_x_mm` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `2.376085379` | `2.376085379` | same |
| `t_cw_x_mm` | `t_cw_x_mm` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `2.133330626` | `2.133330626` | same |
| `t_cw_x_mm` | `t_cw_x_mm` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `1.60249009` | `1.60249009` | same |
| `t_cw_x_mm` | `t_cw_x_mm` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `2.041472179` | `2.041472179` | same |
| `t_cw_y_mm` | `t_cw_y_mm` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `167.3422381` | `167.3422381` | same |
| `t_cw_y_mm` | `t_cw_y_mm` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `167.3164248` | `167.3164248` | same |
| `t_cw_y_mm` | `t_cw_y_mm` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `167.0889691` | `167.0889691` | same |
| `t_cw_y_mm` | `t_cw_y_mm` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `165.8378993` | `165.8378993` | same |
| `t_cw_y_mm` | `t_cw_y_mm` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `165.9951918` | `165.9951918` | same |
| `t_cw_z_mm` | `t_cw_z_mm` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `268.3166932` | `268.3166932` | same |
| `t_cw_z_mm` | `t_cw_z_mm` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `268.2885955` | `268.2885955` | same |
| `t_cw_z_mm` | `t_cw_z_mm` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `268.4932157` | `268.4932157` | same |
| `t_cw_z_mm` | `t_cw_z_mm` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `269.3412474` | `269.3412474` | same |
| `t_cw_z_mm` | `t_cw_z_mm` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `269.2377322` | `269.2377322` | same |
| `camera_rotation_c_from_w_00` | `scene_13` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `0.04936912318` | `0.04936912318` | same |
| `camera_rotation_c_from_w_00` | `scene_13` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `0.04971880424` | `0.04971880424` | same |
| `camera_rotation_c_from_w_00` | `scene_13` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `0.05021207871` | `0.05021207871` | same |
| `camera_rotation_c_from_w_00` | `scene_13` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `0.05162129545` | `0.05162129545` | same |
| `camera_rotation_c_from_w_00` | `scene_13` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `0.05071621381` | `0.05071621381` | same |
| `camera_rotation_c_from_w_01` | `scene_14` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `0.9983735583` | `0.9983735583` | same |
| `camera_rotation_c_from_w_01` | `scene_14` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `0.9983794049` | `0.9983794049` | same |
| `camera_rotation_c_from_w_01` | `scene_14` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `0.9983431285` | `0.9983431285` | same |
| `camera_rotation_c_from_w_01` | `scene_14` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `0.9982771093` | `0.9982771093` | same |
| `camera_rotation_c_from_w_01` | `scene_14` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `0.9983476519` | `0.9983476519` | same |
| `camera_rotation_c_from_w_02` | `scene_15` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `0.02851398734` | `0.02851398734` | same |
| `camera_rotation_c_from_w_02` | `scene_15` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `0.02768207089` | `0.02768207089` | same |
| `camera_rotation_c_from_w_02` | `scene_15` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `0.02810314287` | `0.02810314287` | same |
| `camera_rotation_c_from_w_02` | `scene_15` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `0.02789468466` | `0.02789468466` | same |
| `camera_rotation_c_from_w_02` | `scene_15` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `0.0270231545` | `0.0270231545` | same |
| `camera_rotation_c_from_w_10` | `scene_16` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `-0.3997199` | `-0.3997199` | same |
| `camera_rotation_c_from_w_10` | `scene_16` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `-0.399691389` | `-0.399691389` | same |
| `camera_rotation_c_from_w_10` | `scene_16` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `-0.3989502413` | `-0.3989502413` | same |
| `camera_rotation_c_from_w_10` | `scene_16` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `-0.3955167283` | `-0.3955167283` | same |
| `camera_rotation_c_from_w_10` | `scene_16` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `-0.3958919115` | `-0.3958919115` | same |
| `camera_rotation_c_from_w_11` | `scene_17` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `0.04591229894` | `0.04591229894` | same |
| `camera_rotation_c_from_w_11` | `scene_17` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `0.04528924685` | `0.04528924685` | same |
| `camera_rotation_c_from_w_11` | `scene_17` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `0.04584553115` | `0.04584553115` | same |
| `camera_rotation_c_from_w_11` | `scene_17` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `0.04608427775` | `0.04608427775` | same |
| `camera_rotation_c_from_w_11` | `scene_17` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `0.04493871929` | `0.04493871929` | same |
| `camera_rotation_c_from_w_12` | `scene_18` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `-0.9154872005` | `-0.9154872005` | same |
| `camera_rotation_c_from_w_12` | `scene_18` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `-0.9155308078` | `-0.9155308078` | same |
| `camera_rotation_c_from_w_12` | `scene_18` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `-0.915825405` | `-0.915825405` | same |
| `camera_rotation_c_from_w_12` | `scene_18` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `-0.917301504` | `-0.917301504` | same |
| `camera_rotation_c_from_w_12` | `scene_18` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `-0.9171964281` | `-0.9171964281` | same |
| `camera_rotation_c_from_w_20` | `scene_19` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `-0.9153064318` | `-0.9153064318` | same |
| `camera_rotation_c_from_w_20` | `scene_19` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `-0.9153004924` | `-0.9153004924` | same |
| `camera_rotation_c_from_w_20` | `scene_19` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `-0.9155967497` | `-0.9155967497` | same |
| `camera_rotation_c_from_w_20` | `scene_19` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `-0.9170066968` | `-0.9170066968` | same |
| `camera_rotation_c_from_w_20` | `scene_19` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `-0.9168952795` | `-0.9168952795` | same |
| `camera_rotation_c_from_w_21` | `scene_20` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `0.03379992445` | `0.03379992445` | same |
| `camera_rotation_c_from_w_21` | `scene_20` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `0.03445476863` | `0.03445476863` | same |
| `camera_rotation_c_from_w_21` | `scene_20` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `0.0347729805` | `0.0347729805` | same |
| `camera_rotation_c_from_w_21` | `scene_20` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `0.03631922094` | `0.03631922094` | same |
| `camera_rotation_c_from_w_21` | `scene_20` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `0.03581893547` | `0.03581893547` | same |
| `camera_rotation_c_from_w_22` | `scene_21` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `0.4013365585` | `0.4013365585` | same |
| `camera_rotation_c_from_w_22` | `scene_21` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `0.4012950926` | `0.4012950926` | same |
| `camera_rotation_c_from_w_22` | `scene_21` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `0.4005910423` | `0.4005910423` | same |
| `camera_rotation_c_from_w_22` | `scene_21` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `0.3972142592` | `0.3972142592` | same |
| `camera_rotation_c_from_w_22` | `scene_21` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `0.3975167773` | `0.3975167773` | same |
| `camera_rotation_w_from_c_00` | `transpose(scene_13..21)[0,0]` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `0.04936912318` | `0.04936912318` | same |
| `camera_rotation_w_from_c_00` | `transpose(scene_13..21)[0,0]` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `0.04971880424` | `0.04971880424` | same |
| `camera_rotation_w_from_c_00` | `transpose(scene_13..21)[0,0]` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `0.05021207871` | `0.05021207871` | same |
| `camera_rotation_w_from_c_00` | `transpose(scene_13..21)[0,0]` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `0.05162129545` | `0.05162129545` | same |
| `camera_rotation_w_from_c_00` | `transpose(scene_13..21)[0,0]` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `0.05071621381` | `0.05071621381` | same |
| `camera_rotation_w_from_c_01` | `transpose(scene_13..21)[0,1]` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `-0.3997199` | `-0.3997199` | same |
| `camera_rotation_w_from_c_01` | `transpose(scene_13..21)[0,1]` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `-0.399691389` | `-0.399691389` | same |
| `camera_rotation_w_from_c_01` | `transpose(scene_13..21)[0,1]` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `-0.3989502413` | `-0.3989502413` | same |
| `camera_rotation_w_from_c_01` | `transpose(scene_13..21)[0,1]` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `-0.3955167283` | `-0.3955167283` | same |
| `camera_rotation_w_from_c_01` | `transpose(scene_13..21)[0,1]` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `-0.3958919115` | `-0.3958919115` | same |
| `camera_rotation_w_from_c_02` | `transpose(scene_13..21)[0,2]` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `-0.9153064318` | `-0.9153064318` | same |
| `camera_rotation_w_from_c_02` | `transpose(scene_13..21)[0,2]` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `-0.9153004924` | `-0.9153004924` | same |
| `camera_rotation_w_from_c_02` | `transpose(scene_13..21)[0,2]` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `-0.9155967497` | `-0.9155967497` | same |
| `camera_rotation_w_from_c_02` | `transpose(scene_13..21)[0,2]` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `-0.9170066968` | `-0.9170066968` | same |
| `camera_rotation_w_from_c_02` | `transpose(scene_13..21)[0,2]` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `-0.9168952795` | `-0.9168952795` | same |
| `camera_rotation_w_from_c_10` | `transpose(scene_13..21)[1,0]` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `0.9983735583` | `0.9983735583` | same |
| `camera_rotation_w_from_c_10` | `transpose(scene_13..21)[1,0]` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `0.9983794049` | `0.9983794049` | same |
| `camera_rotation_w_from_c_10` | `transpose(scene_13..21)[1,0]` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `0.9983431285` | `0.9983431285` | same |
| `camera_rotation_w_from_c_10` | `transpose(scene_13..21)[1,0]` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `0.9982771093` | `0.9982771093` | same |
| `camera_rotation_w_from_c_10` | `transpose(scene_13..21)[1,0]` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `0.9983476519` | `0.9983476519` | same |
| `camera_rotation_w_from_c_11` | `transpose(scene_13..21)[1,1]` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `0.04591229894` | `0.04591229894` | same |
| `camera_rotation_w_from_c_11` | `transpose(scene_13..21)[1,1]` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `0.04528924685` | `0.04528924685` | same |
| `camera_rotation_w_from_c_11` | `transpose(scene_13..21)[1,1]` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `0.04584553115` | `0.04584553115` | same |
| `camera_rotation_w_from_c_11` | `transpose(scene_13..21)[1,1]` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `0.04608427775` | `0.04608427775` | same |
| `camera_rotation_w_from_c_11` | `transpose(scene_13..21)[1,1]` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `0.04493871929` | `0.04493871929` | same |
| `camera_rotation_w_from_c_12` | `transpose(scene_13..21)[1,2]` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `0.03379992445` | `0.03379992445` | same |
| `camera_rotation_w_from_c_12` | `transpose(scene_13..21)[1,2]` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `0.03445476863` | `0.03445476863` | same |
| `camera_rotation_w_from_c_12` | `transpose(scene_13..21)[1,2]` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `0.0347729805` | `0.0347729805` | same |
| `camera_rotation_w_from_c_12` | `transpose(scene_13..21)[1,2]` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `0.03631922094` | `0.03631922094` | same |
| `camera_rotation_w_from_c_12` | `transpose(scene_13..21)[1,2]` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `0.03581893547` | `0.03581893547` | same |
| `camera_rotation_w_from_c_20` | `transpose(scene_13..21)[2,0]` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `0.02851398734` | `0.02851398734` | same |
| `camera_rotation_w_from_c_20` | `transpose(scene_13..21)[2,0]` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `0.02768207089` | `0.02768207089` | same |
| `camera_rotation_w_from_c_20` | `transpose(scene_13..21)[2,0]` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `0.02810314287` | `0.02810314287` | same |
| `camera_rotation_w_from_c_20` | `transpose(scene_13..21)[2,0]` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `0.02789468466` | `0.02789468466` | same |
| `camera_rotation_w_from_c_20` | `transpose(scene_13..21)[2,0]` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `0.0270231545` | `0.0270231545` | same |
| `camera_rotation_w_from_c_21` | `transpose(scene_13..21)[2,1]` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `-0.9154872005` | `-0.9154872005` | same |
| `camera_rotation_w_from_c_21` | `transpose(scene_13..21)[2,1]` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `-0.9155308078` | `-0.9155308078` | same |
| `camera_rotation_w_from_c_21` | `transpose(scene_13..21)[2,1]` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `-0.915825405` | `-0.915825405` | same |
| `camera_rotation_w_from_c_21` | `transpose(scene_13..21)[2,1]` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `-0.917301504` | `-0.917301504` | same |
| `camera_rotation_w_from_c_21` | `transpose(scene_13..21)[2,1]` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `-0.9171964281` | `-0.9171964281` | same |
| `camera_rotation_w_from_c_22` | `transpose(scene_13..21)[2,2]` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `0.4013365585` | `0.4013365585` | same |
| `camera_rotation_w_from_c_22` | `transpose(scene_13..21)[2,2]` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `0.4012950926` | `0.4012950926` | same |
| `camera_rotation_w_from_c_22` | `transpose(scene_13..21)[2,2]` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `0.4005910423` | `0.4005910423` | same |
| `camera_rotation_w_from_c_22` | `transpose(scene_13..21)[2,2]` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `0.3972142592` | `0.3972142592` | same |
| `camera_rotation_w_from_c_22` | `transpose(scene_13..21)[2,2]` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `0.3975167773` | `0.3975167773` | same |
| `table_u_axis_camera_x` | `scene_07` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `0.04936912318` | `0.04936912318` | same |
| `table_u_axis_camera_x` | `scene_07` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `0.04971880424` | `0.04971880424` | same |
| `table_u_axis_camera_x` | `scene_07` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `0.05021207871` | `0.05021207871` | same |
| `table_u_axis_camera_x` | `scene_07` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `0.05162129545` | `0.05162129545` | same |
| `table_u_axis_camera_x` | `scene_07` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `0.05071621381` | `0.05071621381` | same |
| `table_u_axis_camera_y` | `scene_08` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `-0.3997199` | `-0.3997199` | same |
| `table_u_axis_camera_y` | `scene_08` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `-0.399691389` | `-0.399691389` | same |
| `table_u_axis_camera_y` | `scene_08` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `-0.3989502413` | `-0.3989502413` | same |
| `table_u_axis_camera_y` | `scene_08` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `-0.3955167283` | `-0.3955167283` | same |
| `table_u_axis_camera_y` | `scene_08` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `-0.3958919115` | `-0.3958919115` | same |
| `table_u_axis_camera_z` | `scene_09` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `-0.9153064318` | `-0.9153064318` | same |
| `table_u_axis_camera_z` | `scene_09` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `-0.9153004924` | `-0.9153004924` | same |
| `table_u_axis_camera_z` | `scene_09` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `-0.9155967497` | `-0.9155967497` | same |
| `table_u_axis_camera_z` | `scene_09` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `-0.9170066968` | `-0.9170066968` | same |
| `table_u_axis_camera_z` | `scene_09` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `-0.9168952795` | `-0.9168952795` | same |
| `table_v_axis_camera_x` | `scene_10` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `0.9983735583` | `0.9983735583` | same |
| `table_v_axis_camera_x` | `scene_10` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `0.9983794049` | `0.9983794049` | same |
| `table_v_axis_camera_x` | `scene_10` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `0.9983431285` | `0.9983431285` | same |
| `table_v_axis_camera_x` | `scene_10` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `0.9982771093` | `0.9982771093` | same |
| `table_v_axis_camera_x` | `scene_10` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `0.9983476519` | `0.9983476519` | same |
| `table_v_axis_camera_y` | `scene_11` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `0.04591229894` | `0.04591229894` | same |
| `table_v_axis_camera_y` | `scene_11` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `0.04528924685` | `0.04528924685` | same |
| `table_v_axis_camera_y` | `scene_11` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `0.04584553115` | `0.04584553115` | same |
| `table_v_axis_camera_y` | `scene_11` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `0.04608427775` | `0.04608427775` | same |
| `table_v_axis_camera_y` | `scene_11` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `0.04493871929` | `0.04493871929` | same |
| `table_v_axis_camera_z` | `scene_12` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `0.03379992445` | `0.03379992445` | same |
| `table_v_axis_camera_z` | `scene_12` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `0.03445476863` | `0.03445476863` | same |
| `table_v_axis_camera_z` | `scene_12` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `0.0347729805` | `0.0347729805` | same |
| `table_v_axis_camera_z` | `scene_12` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `0.03631922094` | `0.03631922094` | same |
| `table_v_axis_camera_z` | `scene_12` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `0.03581893547` | `0.03581893547` | same |
| `table_distance_camera_mm` | `scene_03` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `-30.21323226` | `-30.21323226` | same |
| `table_distance_camera_mm` | `scene_03` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `-30.29990244` | `-30.29990244` | same |
| `table_distance_camera_mm` | `scene_03` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `-30.35599338` | `-30.35599338` | same |
| `table_distance_camera_mm` | `scene_03` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `-30.03940574` | `-30.03940574` | same |
| `table_distance_camera_mm` | `scene_03` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `-29.99534839` | `-29.99534839` | same |

## Stage 2B metric pose CSV vs 旧 iris65/depth_priors CSV

- 新 CSV 行数：583
- 旧对照 CSV 行数：583
- 本节抽样：前 5 个在旧文件中也存在的样本
- 样本：`dataset_dual_rigid_body_11/img_20260805_114114_00000`, `dataset_dual_rigid_body_11/img_20260805_114116_00001`, `dataset_dual_rigid_body_11/img_20260805_114122_00004`, `dataset_dual_rigid_body_11/img_20260805_114125_00008`, `dataset_dual_rigid_body_11/img_20260805_114127_00009`
- 抽样字段判断正常：3 个；出现 different：30 个；旧值缺失：0 个。
- 出现 different 的新字段：`metric_pose_status`, `reason`, `left_iris_proxy_camera_x_mm`, `left_iris_proxy_camera_y_mm`, `left_iris_proxy_camera_z_mm`, `right_iris_proxy_camera_x_mm`, `right_iris_proxy_camera_y_mm`, `right_iris_proxy_camera_z_mm`, `iris_midpoint_camera_x_mm`, `iris_midpoint_camera_y_mm`, `iris_midpoint_camera_z_mm`, `head_rotation_c_from_h_00`, `head_rotation_c_from_h_01`, `head_rotation_c_from_h_02`, `head_rotation_c_from_h_10`, `head_rotation_c_from_h_11`, `head_rotation_c_from_h_12`, `head_rotation_c_from_h_20`, `head_rotation_c_from_h_21`, `head_rotation_c_from_h_22`, `ipd_mm`, `iris_baseline_residual_mm`, `iris_ray_system_condition`, `rotation_pnp_reprojection_error_mean_px`, `rotation_pnp_reprojection_error_max_px`, `rotation_pnp_confidence`, `rotation_pnp_solver_method`, `rotation_pnp_geometry_version`, `rotation_pnp_candidate_count`, `fixed_shape_calibration_count`
- 本节只覆盖旧 Iris65 depth prior 中有同语义来源的字段。
- 新 CSV 中旧文件没有直接等价字段：`head_origin_camera_x_mm`, `head_origin_camera_y_mm`, `head_origin_camera_z_mm`, `eye_proxy_midpoint_head_x_mm`, `eye_proxy_midpoint_head_y_mm`, `eye_proxy_midpoint_head_z_mm`, `flame_scale_from_ipd_mm_per_unit`, `rotation_pnp_min_normalized_object_depth`, `eye15_quality_score`。其中 head_origin_* 是新流程为虚拟相机中心显式新增的 t_C_H。

| 新字段 | 旧字段/来源 | 样本 | 新值 | 旧值 | 判断 |
|---|---|---|---|---|---|
| `sample_id` | `sample_id` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | same |
| `sample_id` | `sample_id` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | same |
| `sample_id` | `sample_id` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | same |
| `sample_id` | `sample_id` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | same |
| `sample_id` | `sample_id` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | same |
| `dataset` | `dataset` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `dataset_dual_rigid_body_11` | `dataset_dual_rigid_body_11` | same |
| `dataset` | `dataset` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `dataset_dual_rigid_body_11` | `dataset_dual_rigid_body_11` | same |
| `dataset` | `dataset` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `dataset_dual_rigid_body_11` | `dataset_dual_rigid_body_11` | same |
| `dataset` | `dataset` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `dataset_dual_rigid_body_11` | `dataset_dual_rigid_body_11` | same |
| `dataset` | `dataset` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `dataset_dual_rigid_body_11` | `dataset_dual_rigid_body_11` | same |
| `image_name` | `image_name` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `img_20260805_114114_00000.jpg` | `img_20260805_114114_00000.jpg` | same |
| `image_name` | `image_name` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `img_20260805_114116_00001.jpg` | `img_20260805_114116_00001.jpg` | same |
| `image_name` | `image_name` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `img_20260805_114122_00004.jpg` | `img_20260805_114122_00004.jpg` | same |
| `image_name` | `image_name` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `img_20260805_114125_00008.jpg` | `img_20260805_114125_00008.jpg` | same |
| `image_name` | `image_name` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `img_20260805_114127_00009.jpg` | `img_20260805_114127_00009.jpg` | same |
| `metric_pose_status` | `depth_prior_status` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `success` | `failed` | different |
| `metric_pose_status` | `depth_prior_status` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `success` | `success` | same |
| `metric_pose_status` | `depth_prior_status` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `success` | `success` | same |
| `metric_pose_status` | `depth_prior_status` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `success` | `success` | same |
| `metric_pose_status` | `depth_prior_status` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `success` | `success` | same |
| `reason` | `reason` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `` | `ValueError: Iris/IPD baseline residual 4.12mm exceeds 4.00mm.` | different |
| `reason` | `reason` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `` | `` | same |
| `reason` | `reason` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `` | `` | same |
| `reason` | `reason` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `` | `` | same |
| `reason` | `reason` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `` | `` | same |
| `left_iris_proxy_camera_x_mm` | `left_iris_camera_x_mm` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `120.1269601908648` | `` | different |
| `left_iris_proxy_camera_x_mm` | `left_iris_camera_x_mm` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `118.7561966144223` | `116.8852686931273` | different |
| `left_iris_proxy_camera_x_mm` | `left_iris_camera_x_mm` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `118.084294299951` | `118.92803734898347` | different |
| `left_iris_proxy_camera_x_mm` | `left_iris_camera_x_mm` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `117.06412552562139` | `118.79354220279113` | different |
| `left_iris_proxy_camera_x_mm` | `left_iris_camera_x_mm` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `124.51364080363624` | `122.58253811112729` | different |
| `left_iris_proxy_camera_y_mm` | `left_iris_camera_y_mm` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `-77.5588744745316` | `` | different |
| `left_iris_proxy_camera_y_mm` | `left_iris_camera_y_mm` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `-75.89306224522043` | `-74.92210995545439` | different |
| `left_iris_proxy_camera_y_mm` | `left_iris_camera_y_mm` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `-69.50932115155888` | `-70.05663836113585` | different |
| `left_iris_proxy_camera_y_mm` | `left_iris_camera_y_mm` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `-67.53770442440188` | `-68.5427896632066` | different |
| `left_iris_proxy_camera_y_mm` | `left_iris_camera_y_mm` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `-67.56883952941214` | `-66.48254343382487` | different |
| `left_iris_proxy_camera_z_mm` | `left_iris_camera_z_mm` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `648.9106414604748` | `` | different |
| `left_iris_proxy_camera_z_mm` | `left_iris_camera_z_mm` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `646.6774028449652` | `635.6110562674991` | different |
| `left_iris_proxy_camera_z_mm` | `left_iris_camera_z_mm` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `632.7733447039825` | `637.4765863511094` | different |
| `left_iris_proxy_camera_z_mm` | `left_iris_camera_z_mm` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `630.4350535028784` | `639.9711341540891` | different |
| `left_iris_proxy_camera_z_mm` | `left_iris_camera_z_mm` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `647.2547379373734` | `636.1866871873455` | different |
| `right_iris_proxy_camera_x_mm` | `right_iris_camera_x_mm` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `57.5059849616479` | `` | different |
| `right_iris_proxy_camera_x_mm` | `right_iris_camera_x_mm` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `56.26202559646426` | `54.936338636301564` | different |
| `right_iris_proxy_camera_x_mm` | `right_iris_camera_x_mm` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `56.08329151104062` | `56.574377281499466` | different |
| `right_iris_proxy_camera_x_mm` | `right_iris_camera_x_mm` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `55.416056613865806` | `56.51086697174922` | different |
| `right_iris_proxy_camera_x_mm` | `right_iris_camera_x_mm` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `63.0781886714929` | `61.629087094818026` | different |
| `right_iris_proxy_camera_y_mm` | `right_iris_camera_y_mm` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `-85.00854390327805` | `` | different |
| `right_iris_proxy_camera_y_mm` | `right_iris_camera_y_mm` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `-83.93534046427574` | `-82.1932427859827` | different |
| `right_iris_proxy_camera_y_mm` | `right_iris_camera_y_mm` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `-79.0448172176019` | `-79.8552270440409` | different |
| `right_iris_proxy_camera_y_mm` | `right_iris_camera_y_mm` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `-77.9937529499314` | `-79.52635595460909` | different |
| `right_iris_proxy_camera_y_mm` | `right_iris_camera_y_mm` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `-79.33509288314538` | `-77.76596683152799` | different |
| `right_iris_proxy_camera_z_mm` | `right_iris_camera_z_mm` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `633.3705587715509` | `` | different |
| `right_iris_proxy_camera_z_mm` | `right_iris_camera_z_mm` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `630.8530730934093` | `617.636472323419` | different |
| `right_iris_proxy_camera_z_mm` | `right_iris_camera_z_mm` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `615.9203688743697` | `622.2554399203422` | different |
| `right_iris_proxy_camera_z_mm` | `right_iris_camera_z_mm` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `612.8878818941137` | `625.2991541884209` | different |
| `right_iris_proxy_camera_z_mm` | `right_iris_camera_z_mm` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `629.7580854436803` | `616.9619706034804` | different |
| `iris_midpoint_camera_x_mm` | `iris_midpoint_camera_x_mm` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `88.81647257625636` | `` | different |
| `iris_midpoint_camera_x_mm` | `iris_midpoint_camera_x_mm` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `87.50911110544328` | `85.91080366471444` | different |
| `iris_midpoint_camera_x_mm` | `iris_midpoint_camera_x_mm` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `87.08379290549581` | `87.75120731524146` | different |
| `iris_midpoint_camera_x_mm` | `iris_midpoint_camera_x_mm` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `86.2400910697436` | `87.65220458727018` | different |
| `iris_midpoint_camera_x_mm` | `iris_midpoint_camera_x_mm` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `93.79591473756457` | `92.10581260297266` | different |
| `iris_midpoint_camera_y_mm` | `iris_midpoint_camera_y_mm` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `-81.28370918890482` | `` | different |
| `iris_midpoint_camera_y_mm` | `iris_midpoint_camera_y_mm` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `-79.91420135474809` | `-78.55767637071855` | different |
| `iris_midpoint_camera_y_mm` | `iris_midpoint_camera_y_mm` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `-74.2770691845804` | `-74.95593270258837` | different |
| `iris_midpoint_camera_y_mm` | `iris_midpoint_camera_y_mm` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `-72.76572868716664` | `-74.03457280890785` | different |
| `iris_midpoint_camera_y_mm` | `iris_midpoint_camera_y_mm` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `-73.45196620627877` | `-72.12425513267644` | different |
| `iris_midpoint_camera_z_mm` | `iris_midpoint_depth_z_mm` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `641.1406001160128` | `` | different |
| `iris_midpoint_camera_z_mm` | `iris_midpoint_depth_z_mm` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `638.7652379691872` | `626.6237642954591` | different |
| `iris_midpoint_camera_z_mm` | `iris_midpoint_depth_z_mm` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `624.3468567891762` | `629.8660131357258` | different |
| `iris_midpoint_camera_z_mm` | `iris_midpoint_depth_z_mm` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `621.661467698496` | `632.635144171255` | different |
| `iris_midpoint_camera_z_mm` | `iris_midpoint_depth_z_mm` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `638.5064116905269` | `626.5743288954129` | different |
| `head_rotation_c_from_h_00` | `pnp_rotation_00` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `0.9664366447363417` | `` | different |
| `head_rotation_c_from_h_00` | `pnp_rotation_00` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `0.9636279381135144` | `0.9574380743213871` | different |
| `head_rotation_c_from_h_00` | `pnp_rotation_00` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `0.958094493217372` | `0.9652955083502891` | different |
| `head_rotation_c_from_h_00` | `pnp_rotation_00` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `0.9540768561212384` | `0.9655319408080427` | different |
| `head_rotation_c_from_h_00` | `pnp_rotation_00` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `0.9508963846097109` | `0.9465102818872485` | different |
| `head_rotation_c_from_h_01` | `pnp_rotation_01` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `0.11779292475702668` | `` | different |
| `head_rotation_c_from_h_01` | `pnp_rotation_01` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `0.13157147677856995` | `0.12696574636047941` | different |
| `head_rotation_c_from_h_01` | `pnp_rotation_01` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `0.17195222275474778` | `0.16491269081091658` | different |
| `head_rotation_c_from_h_01` | `pnp_rotation_01` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `0.18574488256655447` | `0.18124913520447664` | different |
| `head_rotation_c_from_h_01` | `pnp_rotation_01` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `0.20425776564334874` | `0.1953096399298377` | different |
| `head_rotation_c_from_h_02` | `pnp_rotation_02` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `0.22830908564476438` | `` | different |
| `head_rotation_c_from_h_02` | `pnp_rotation_02` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `0.23261587088030558` | `0.25921426097127037` | different |
| `head_rotation_c_from_h_02` | `pnp_rotation_02` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `0.22910123342367228` | `0.20250527392702622` | different |
| `head_rotation_c_from_h_02` | `pnp_rotation_02` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `0.2350238098877434` | `0.18680691172192712` | different |
| `head_rotation_c_from_h_02` | `pnp_rotation_02` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `0.23253995551424564` | `0.25685098954880037` | different |
| `head_rotation_c_from_h_10` | `pnp_rotation_10` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `0.07257911292257238` | `` | different |
| `head_rotation_c_from_h_10` | `pnp_rotation_10` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `0.08918517677119525` | `0.057964477765939625` | different |
| `head_rotation_c_from_h_10` | `pnp_rotation_10` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `0.10653071696692293` | `0.10196794028789602` | different |
| `head_rotation_c_from_h_10` | `pnp_rotation_10` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `0.1170650359821757` | `0.11889646483960761` | different |
| `head_rotation_c_from_h_10` | `pnp_rotation_10` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `0.14111118123950456` | `0.11784366928189477` | different |
| `head_rotation_c_from_h_11` | `pnp_rotation_11` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `-0.9776751346354441` | `` | different |
| `head_rotation_c_from_h_11` | `pnp_rotation_11` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `-0.9788360229252672` | `-0.9643391187067124` | different |
| `head_rotation_c_from_h_11` | `pnp_rotation_11` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `-0.9563106689378158` | `-0.9518538276303681` | different |
| `head_rotation_c_from_h_11` | `pnp_rotation_11` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `-0.9533530790430289` | `-0.9455729922076134` | different |
| `head_rotation_c_from_h_11` | `pnp_rotation_11` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `-0.9547940255544123` | `-0.9502564470018521` | different |
| `head_rotation_c_from_h_12` | `pnp_rotation_12` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `0.19718925803131016` | `` | different |
| `head_rotation_c_from_h_12` | `pnp_rotation_12` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `0.1841902398829438` | `0.25824442578550094` | different |
| `head_rotation_c_from_h_12` | `pnp_rotation_12` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `0.2722519252791808` | `0.28909657552254336` | different |
| `head_rotation_c_from_h_12` | `pnp_rotation_12` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `0.2782331468924367` | `0.3029114508502189` | different |
| `head_rotation_c_from_h_12` | `pnp_rotation_12` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `0.26164098168060773` | `0.2883150265622582` | different |
| `head_rotation_c_from_h_20` | `pnp_rotation_20` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `0.2464396154804162` | `` | different |
| `head_rotation_c_from_h_20` | `pnp_rotation_20` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `0.25192697579137374` | `0.28275864824453617` | different |
| `head_rotation_c_from_h_20` | `pnp_rotation_20` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `0.26590627749088536` | `0.24043111427642005` | different |
| `head_rotation_c_from_h_20` | `pnp_rotation_20` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `0.2757410560005622` | `0.23154200899210162` | different |
| `head_rotation_c_from_h_20` | `pnp_rotation_20` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `0.2754699625465038` | `0.3003850127617891` | different |
| `head_rotation_c_from_h_21` | `pnp_rotation_21` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `-0.17400045400156744` | `` | different |
| `head_rotation_c_from_h_21` | `pnp_rotation_21` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `-0.15674497351478917` | `-0.23222782646161852` | different |
| `head_rotation_c_from_h_21` | `pnp_rotation_21` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `-0.2364367517231805` | `-0.2584145801515862` | different |
| `head_rotation_c_from_h_21` | `pnp_rotation_21` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `-0.2379427352946781` | `-0.27025999962105013` | different |
| `head_rotation_c_from_h_21` | `pnp_rotation_21` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `-0.21597947573782847` | `-0.2426248739966564` | different |
| `head_rotation_c_from_h_22` | `pnp_rotation_22` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `-0.9534103827466497` | `` | different |
| `head_rotation_c_from_h_22` | `pnp_rotation_22` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `-0.954967963937252` | `-0.9306544919892855` | different |
| `head_rotation_c_from_h_22` | `pnp_rotation_22` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `-0.9345541792884623` | `-0.9356360318269484` | different |
| `head_rotation_c_from_h_22` | `pnp_rotation_22` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `-0.9313163397880344` | `-0.9345308077729335` | different |
| `head_rotation_c_from_h_22` | `pnp_rotation_22` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `-0.9367332415339175` | `-0.9224435021323563` | different |
| `ipd_mm` | `measured_ipd_mm` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `65.0` | `` | different |
| `ipd_mm` | `measured_ipd_mm` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `65.0` | `65.0` | same |
| `ipd_mm` | `measured_ipd_mm` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `65.0` | `65.0` | same |
| `ipd_mm` | `measured_ipd_mm` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `65.0` | `65.0` | same |
| `ipd_mm` | `measured_ipd_mm` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `65.0` | `65.0` | same |
| `iris_baseline_residual_mm` | `iris_baseline_residual_mm` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `2.5732696644032576` | `` | different |
| `iris_baseline_residual_mm` | `iris_baseline_residual_mm` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `2.095456451377987` | `3.372687980325617` | different |
| `iris_baseline_residual_mm` | `iris_baseline_residual_mm` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `2.434578737788804` | `3.053757905448658` | different |
| `iris_baseline_residual_mm` | `iris_baseline_residual_mm` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `2.698780690537835` | `3.156365231427804` | different |
| `iris_baseline_residual_mm` | `iris_baseline_residual_mm` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `2.471367000189979` | `3.573714860005076` | different |
| `iris_ray_system_condition` | `iris_ray_system_condition` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `21.456063186170184` | `` | different |
| `iris_ray_system_condition` | `iris_ray_system_condition` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `21.376779560247414` | `21.289042660962988` | different |
| `iris_ray_system_condition` | `iris_ray_system_condition` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `21.020895793007533` | `21.00490405989671` | different |
| `iris_ray_system_condition` | `iris_ray_system_condition` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `20.99838105341949` | `21.004728568282047` | different |
| `iris_ray_system_condition` | `iris_ray_system_condition` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `21.622663887210408` | `21.49676883080341` | different |
| `rotation_pnp_reprojection_error_mean_px` | `pnp_reprojection_error_mean_px` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `4.325973943218245` | `` | different |
| `rotation_pnp_reprojection_error_mean_px` | `pnp_reprojection_error_mean_px` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `3.4109920017695714` | `4.071202324560385` | different |
| `rotation_pnp_reprojection_error_mean_px` | `pnp_reprojection_error_mean_px` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `4.539165403205429` | `4.109367591310979` | different |
| `rotation_pnp_reprojection_error_mean_px` | `pnp_reprojection_error_mean_px` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `4.6628487674845625` | `4.456152493939585` | different |
| `rotation_pnp_reprojection_error_mean_px` | `pnp_reprojection_error_mean_px` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `3.6149383996233997` | `4.2086366688238135` | different |
| `rotation_pnp_reprojection_error_max_px` | `pnp_reprojection_error_max_px` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `5.919223316082468` | `` | different |
| `rotation_pnp_reprojection_error_max_px` | `pnp_reprojection_error_max_px` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `4.697132360305593` | `6.7451042383257604` | different |
| `rotation_pnp_reprojection_error_max_px` | `pnp_reprojection_error_max_px` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `7.257026601837879` | `7.262608568847571` | different |
| `rotation_pnp_reprojection_error_max_px` | `pnp_reprojection_error_max_px` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `7.795266418480173` | `6.417538073497185` | different |
| `rotation_pnp_reprojection_error_max_px` | `pnp_reprojection_error_max_px` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `4.761227678359049` | `7.559887764614501` | different |
| `rotation_pnp_confidence` | `pnp_confidence` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `0.33908872309350596` | `` | different |
| `rotation_pnp_confidence` | `pnp_confidence` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `0.42624200786271177` | `0.29559160755855324` | different |
| `rotation_pnp_confidence` | `pnp_confidence` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `0.32148919279939425` | `0.29270308642264004` | different |
| `rotation_pnp_confidence` | `pnp_confidence` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `0.3117005923358255` | `0.268433293645767` | different |
| `rotation_pnp_confidence` | `pnp_confidence` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `0.405054116462971` | `0.28557947469244094` | different |
| `rotation_pnp_solver_method` | `pnp_solver_method` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `EPNP+ITERATIVE` | `` | different |
| `rotation_pnp_solver_method` | `pnp_solver_method` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `EPNP+ITERATIVE` | `SQPNP` | different |
| `rotation_pnp_solver_method` | `pnp_solver_method` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `SQPNP` | `SQPNP` | same |
| `rotation_pnp_solver_method` | `pnp_solver_method` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `SQPNP` | `SQPNP` | same |
| `rotation_pnp_solver_method` | `pnp_solver_method` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `ITERATIVE` | `SQPNP` | different |
| `rotation_pnp_geometry_version` | `pnp_geometry_version` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `mediapipe_flame_rotation_only_normalized_v1` | `` | different |
| `rotation_pnp_geometry_version` | `pnp_geometry_version` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `mediapipe_flame_rotation_only_normalized_v1` | `mediapipe_flame_head_local_multistart_v2` | different |
| `rotation_pnp_geometry_version` | `pnp_geometry_version` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `mediapipe_flame_rotation_only_normalized_v1` | `mediapipe_flame_head_local_multistart_v2` | different |
| `rotation_pnp_geometry_version` | `pnp_geometry_version` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `mediapipe_flame_rotation_only_normalized_v1` | `mediapipe_flame_head_local_multistart_v2` | different |
| `rotation_pnp_geometry_version` | `pnp_geometry_version` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `mediapipe_flame_rotation_only_normalized_v1` | `mediapipe_flame_head_local_multistart_v2` | different |
| `rotation_pnp_candidate_count` | `pnp_candidate_count` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `5` | `` | different |
| `rotation_pnp_candidate_count` | `pnp_candidate_count` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `5` | `5` | same |
| `rotation_pnp_candidate_count` | `pnp_candidate_count` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `5` | `5` | same |
| `rotation_pnp_candidate_count` | `pnp_candidate_count` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `5` | `5` | same |
| `rotation_pnp_candidate_count` | `pnp_candidate_count` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `5` | `5` | same |
| `fixed_shape_calibration_count` | `fixed_shape_calibration_count` | `dataset_dual_rigid_body_11/img_20260805_114114_00000` | `37` | `` | different |
| `fixed_shape_calibration_count` | `fixed_shape_calibration_count` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `37` | `80` | different |
| `fixed_shape_calibration_count` | `fixed_shape_calibration_count` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `37` | `80` | different |
| `fixed_shape_calibration_count` | `fixed_shape_calibration_count` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `37` | `80` | different |
| `fixed_shape_calibration_count` | `fixed_shape_calibration_count` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `37` | `80` | different |

## Stage 3 final training CSV vs 旧 common/modelv1 + virtual manifest

- 新 CSV 行数：578
- 旧对照 CSV 行数：505
- 本节抽样：前 5 个在旧文件中也存在的样本
- 样本：`dataset_dual_rigid_body_11/img_20260805_114116_00001`, `dataset_dual_rigid_body_11/img_20260805_114122_00004`, `dataset_dual_rigid_body_11/img_20260805_114125_00008`, `dataset_dual_rigid_body_11/img_20260805_114127_00009`, `dataset_dual_rigid_body_11/img_20260805_114129_00013`
- 抽样字段判断正常：8 个；出现 different：8 个；旧值缺失：0 个。
- 出现 different 的新字段：`virtual_left_eye_path`, `virtual_right_eye_path`, `table_frame7_n_0`, `table_frame7_n_1`, `table_frame7_n_2`, `table_frame7_n_3`, `table_frame7_n_4`, `table_frame7_n_5`
- sample/source/UV 字段来自旧 common/modelv1 CSV 对照。
- virtual_face_path 对照旧 manifest 的 normalized_face_path，目录不同但文件名一致即可认为路径语义对上。
- table_frame7_n_0..6 旧文件没有同名字段，这里按旧 loader 逻辑从 scene_07..12、scene_03 和旧 manifest 的 rotation_n_from_c 现场推导后对照。
- 当前新配置 generate_eye_images=false，所以 virtual_left/right_eye_path 为空；旧 manifest 有眼图路径，different 属于预期设计差异。

| 新字段 | 旧字段/来源 | 样本 | 新值 | 旧值 | 判断 |
|---|---|---|---|---|---|
| `sample_id` | `sample_id` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | same |
| `sample_id` | `sample_id` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | same |
| `sample_id` | `sample_id` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | same |
| `sample_id` | `sample_id` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | same |
| `sample_id` | `sample_id` | `dataset_dual_rigid_body_11/img_20260805_114129_00013` | `dataset_dual_rigid_body_11/img_20260805_114129_00013` | `dataset_dual_rigid_body_11/img_20260805_114129_00013` | same |
| `dataset` | `dataset` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `dataset_dual_rigid_body_11` | `dataset_dual_rigid_body_11` | same |
| `dataset` | `dataset` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `dataset_dual_rigid_body_11` | `dataset_dual_rigid_body_11` | same |
| `dataset` | `dataset` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `dataset_dual_rigid_body_11` | `dataset_dual_rigid_body_11` | same |
| `dataset` | `dataset` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `dataset_dual_rigid_body_11` | `dataset_dual_rigid_body_11` | same |
| `dataset` | `dataset` | `dataset_dual_rigid_body_11/img_20260805_114129_00013` | `dataset_dual_rigid_body_11` | `dataset_dual_rigid_body_11` | same |
| `image_name` | `image_name` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `img_20260805_114116_00001.jpg` | `img_20260805_114116_00001.jpg` | same |
| `image_name` | `image_name` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `img_20260805_114122_00004.jpg` | `img_20260805_114122_00004.jpg` | same |
| `image_name` | `image_name` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `img_20260805_114125_00008.jpg` | `img_20260805_114125_00008.jpg` | same |
| `image_name` | `image_name` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `img_20260805_114127_00009.jpg` | `img_20260805_114127_00009.jpg` | same |
| `image_name` | `image_name` | `dataset_dual_rigid_body_11/img_20260805_114129_00013` | `img_20260805_114129_00013.jpg` | `img_20260805_114129_00013.jpg` | same |
| `source_image_path` | `source_image_path` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `D:\GithubCode\CrossGaze-main\CrossGaz ... ace_img\img_20260805_114116_00001.jpg` | `D:\GithubCode\CrossGaze-main\CrossGaz ... ace_img\img_20260805_114116_00001.jpg` | same |
| `source_image_path` | `source_image_path` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `D:\GithubCode\CrossGaze-main\CrossGaz ... ace_img\img_20260805_114122_00004.jpg` | `D:\GithubCode\CrossGaze-main\CrossGaz ... ace_img\img_20260805_114122_00004.jpg` | same |
| `source_image_path` | `source_image_path` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `D:\GithubCode\CrossGaze-main\CrossGaz ... ace_img\img_20260805_114125_00008.jpg` | `D:\GithubCode\CrossGaze-main\CrossGaz ... ace_img\img_20260805_114125_00008.jpg` | same |
| `source_image_path` | `source_image_path` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `D:\GithubCode\CrossGaze-main\CrossGaz ... ace_img\img_20260805_114127_00009.jpg` | `D:\GithubCode\CrossGaze-main\CrossGaz ... ace_img\img_20260805_114127_00009.jpg` | same |
| `source_image_path` | `source_image_path` | `dataset_dual_rigid_body_11/img_20260805_114129_00013` | `D:\GithubCode\CrossGaze-main\CrossGaz ... ace_img\img_20260805_114129_00013.jpg` | `D:\GithubCode\CrossGaze-main\CrossGaz ... ace_img\img_20260805_114129_00013.jpg` | same |
| `virtual_face_path` | `manifest.normalized_face_path` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `D:\GithubCode\ModelV1\data\preprocess ... body_11\img_20260805_114116_00001.png` | `D:\GithubCode\ModelV1\data\preprocess ... body_11\img_20260805_114116_00001.png` | basename_same |
| `virtual_face_path` | `manifest.normalized_face_path` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `D:\GithubCode\ModelV1\data\preprocess ... body_11\img_20260805_114122_00004.png` | `D:\GithubCode\ModelV1\data\preprocess ... body_11\img_20260805_114122_00004.png` | basename_same |
| `virtual_face_path` | `manifest.normalized_face_path` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `D:\GithubCode\ModelV1\data\preprocess ... body_11\img_20260805_114125_00008.png` | `D:\GithubCode\ModelV1\data\preprocess ... body_11\img_20260805_114125_00008.png` | basename_same |
| `virtual_face_path` | `manifest.normalized_face_path` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `D:\GithubCode\ModelV1\data\preprocess ... body_11\img_20260805_114127_00009.png` | `D:\GithubCode\ModelV1\data\preprocess ... body_11\img_20260805_114127_00009.png` | basename_same |
| `virtual_face_path` | `manifest.normalized_face_path` | `dataset_dual_rigid_body_11/img_20260805_114129_00013` | `D:\GithubCode\ModelV1\data\preprocess ... body_11\img_20260805_114129_00013.png` | `D:\GithubCode\ModelV1\data\preprocess ... body_11\img_20260805_114129_00013.png` | basename_same |
| `virtual_left_eye_path` | `manifest.normalized_left_eye_path` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `` | `D:\GithubCode\ModelV1\data\preprocess ... body_11\img_20260805_114116_00001.png` | different |
| `virtual_left_eye_path` | `manifest.normalized_left_eye_path` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `` | `D:\GithubCode\ModelV1\data\preprocess ... body_11\img_20260805_114122_00004.png` | different |
| `virtual_left_eye_path` | `manifest.normalized_left_eye_path` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `` | `D:\GithubCode\ModelV1\data\preprocess ... body_11\img_20260805_114125_00008.png` | different |
| `virtual_left_eye_path` | `manifest.normalized_left_eye_path` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `` | `D:\GithubCode\ModelV1\data\preprocess ... body_11\img_20260805_114127_00009.png` | different |
| `virtual_left_eye_path` | `manifest.normalized_left_eye_path` | `dataset_dual_rigid_body_11/img_20260805_114129_00013` | `` | `D:\GithubCode\ModelV1\data\preprocess ... body_11\img_20260805_114129_00013.png` | different |
| `virtual_right_eye_path` | `manifest.normalized_right_eye_path` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `` | `D:\GithubCode\ModelV1\data\preprocess ... body_11\img_20260805_114116_00001.png` | different |
| `virtual_right_eye_path` | `manifest.normalized_right_eye_path` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `` | `D:\GithubCode\ModelV1\data\preprocess ... body_11\img_20260805_114122_00004.png` | different |
| `virtual_right_eye_path` | `manifest.normalized_right_eye_path` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `` | `D:\GithubCode\ModelV1\data\preprocess ... body_11\img_20260805_114125_00008.png` | different |
| `virtual_right_eye_path` | `manifest.normalized_right_eye_path` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `` | `D:\GithubCode\ModelV1\data\preprocess ... body_11\img_20260805_114127_00009.png` | different |
| `virtual_right_eye_path` | `manifest.normalized_right_eye_path` | `dataset_dual_rigid_body_11/img_20260805_114129_00013` | `` | `D:\GithubCode\ModelV1\data\preprocess ... body_11\img_20260805_114129_00013.png` | different |
| `table_frame7_n_0` | `derived from old scene + manifest [0]` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `0.08821606636047363` | `0.09836182099951976` | different |
| `table_frame7_n_0` | `derived from old scene + manifest [0]` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `0.07945850491523743` | `0.08141374737503512` | different |
| `table_frame7_n_0` | `derived from old scene + manifest [0]` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `0.0743228867650032` | `0.07515332278633602` | different |
| `table_frame7_n_0` | `derived from old scene + manifest [0]` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `0.06760529428720474` | `0.07351112163896995` | different |
| `table_frame7_n_0` | `derived from old scene + manifest [0]` | `dataset_dual_rigid_body_11/img_20260805_114129_00013` | `0.07383207231760025` | `0.0750812080015752` | different |
| `table_frame7_n_1` | `derived from old scene + manifest [1]` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `-0.5012932419776917` | `-0.4994130879618101` | different |
| `table_frame7_n_1` | `derived from old scene + manifest [1]` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `-0.5009603500366211` | `-0.49712693983109163` | different |
| `table_frame7_n_1` | `derived from old scene + manifest [1]` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `-0.4974426329135895` | `-0.49413526324619855` | different |
| `table_frame7_n_1` | `derived from old scene + manifest [1]` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `-0.49879613518714905` | `-0.4940598531938012` | different |
| `table_frame7_n_1` | `derived from old scene + manifest [1]` | `dataset_dual_rigid_body_11/img_20260805_114129_00013` | `-0.4997047483921051` | `-0.49667453326570155` | different |
| `table_frame7_n_2` | `derived from old scene + manifest [2]` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `-0.8607688546180725` | `-0.8607622651264373` | close |
| `table_frame7_n_2` | `derived from old scene + manifest [2]` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `-0.8618149757385254` | `-0.8638498500176058` | different |
| `table_frame7_n_2` | `derived from old scene + manifest [2]` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `-0.8643072247505188` | `-0.8661303840507514` | different |
| `table_frame7_n_2` | `derived from old scene + manifest [2]` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `-0.8640786409378052` | `-0.8663143020643425` | different |
| `table_frame7_n_2` | `derived from old scene + manifest [2]` | `dataset_dual_rigid_body_11/img_20260805_114129_00013` | `-0.8630434274673462` | `-0.864682847221766` | different |
| `table_frame7_n_3` | `derived from old scene + manifest [3]` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `0.9871678352355957` | `0.989262378943303` | different |
| `table_frame7_n_3` | `derived from old scene + manifest [3]` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `0.9854308366775513` | `0.9866504168217248` | different |
| `table_frame7_n_3` | `derived from old scene + manifest [3]` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `0.9842431545257568` | `0.9850444310089148` | different |
| `table_frame7_n_3` | `derived from old scene + manifest [3]` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `0.9799919724464417` | `0.9832712621746847` | different |
| `table_frame7_n_3` | `derived from old scene + manifest [3]` | `dataset_dual_rigid_body_11/img_20260805_114129_00013` | `0.9844310283660889` | `0.985699737475412` | different |
| `table_frame7_n_4` | `derived from old scene + manifest [4]` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `-0.07147640734910965` | `-0.04488290447219111` | different |
| `table_frame7_n_4` | `derived from old scene + manifest [4]` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `-0.09094417840242386` | `-0.08245337140889943` | different |
| `table_frame7_n_4` | `derived from old scene + manifest [4]` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `-0.10285034030675888` | `-0.09826541461165655` | different |
| `table_frame7_n_4` | `derived from old scene + manifest [4]` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `-0.1293027251958847` | `-0.10925506478375265` | different |
| `table_frame7_n_4` | `derived from old scene + manifest [4]` | `dataset_dual_rigid_body_11/img_20260805_114129_00013` | `-0.10190194100141525` | `-0.09422041981363569` | different |
| `table_frame7_n_5` | `derived from old scene + manifest [5]` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `0.1427963674068451` | `0.13908685625767184` | different |
| `table_frame7_n_5` | `derived from old scene + manifest [5]` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `0.1437193751335144` | `0.14043636157088885` | different |
| `table_frame7_n_5` | `derived from old scene + manifest [5]` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `0.14383040368556976` | `0.14153236607686132` | different |
| `table_frame7_n_5` | `derived from old scene + manifest [5]` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `0.15131612122058868` | `0.14574455216018756` | different |
| `table_frame7_n_5` | `derived from old scene + manifest [5]` | `dataset_dual_rigid_body_11/img_20260805_114129_00013` | `0.14321842789649963` | `0.13970979587506008` | different |
| `table_frame7_n_6` | `derived from old scene + manifest [6]` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `-0.030299901962280273` | `-0.030299902440000002` | same |
| `table_frame7_n_6` | `derived from old scene + manifest [6]` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `-0.030355993658304214` | `-0.03035599338` | same |
| `table_frame7_n_6` | `derived from old scene + manifest [6]` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `-0.030039405450224876` | `-0.03003940574` | same |
| `table_frame7_n_6` | `derived from old scene + manifest [6]` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `-0.029995348304510117` | `-0.02999534839` | same |
| `table_frame7_n_6` | `derived from old scene + manifest [6]` | `dataset_dual_rigid_body_11/img_20260805_114129_00013` | `-0.030138831585645676` | `-0.03013883232` | same |
| `uv_gt_u_mm` | `uv_gt_u_mm` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `-91.94431042` | `-91.94431042` | same |
| `uv_gt_u_mm` | `uv_gt_u_mm` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `-92.0335085` | `-92.0335085` | same |
| `uv_gt_u_mm` | `uv_gt_u_mm` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `-92.13032686` | `-92.13032686` | same |
| `uv_gt_u_mm` | `uv_gt_u_mm` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `-110.9464735` | `-110.9464735` | same |
| `uv_gt_u_mm` | `uv_gt_u_mm` | `dataset_dual_rigid_body_11/img_20260805_114129_00013` | `-110.9238084` | `-110.9238084` | same |
| `uv_gt_v_mm` | `uv_gt_v_mm` | `dataset_dual_rigid_body_11/img_20260805_114116_00001` | `263.0778118` | `263.0778118` | same |
| `uv_gt_v_mm` | `uv_gt_v_mm` | `dataset_dual_rigid_body_11/img_20260805_114122_00004` | `262.8666208` | `262.8666208` | same |
| `uv_gt_v_mm` | `uv_gt_v_mm` | `dataset_dual_rigid_body_11/img_20260805_114125_00008` | `262.7871002` | `262.7871002` | same |
| `uv_gt_v_mm` | `uv_gt_v_mm` | `dataset_dual_rigid_body_11/img_20260805_114127_00009` | `264.0870756` | `264.0870756` | same |
| `uv_gt_v_mm` | `uv_gt_v_mm` | `dataset_dual_rigid_body_11/img_20260805_114129_00013` | `263.9691408` | `263.9691408` | same |

