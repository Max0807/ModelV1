# 3DGazeNet-main / ModelV1 Context Memory

This file preserves context imported from Codex tasks in `D:\GithubCode\3DGazeNet-main`:

- `Final Model V1` (`019f5c23-a293-7fe2-9a6d-f12f1aa91c3b`)
- `Data Search` (`019f5fd1-a262-7ff3-85c9-e1245c7540fa`)

## Project Goal

The target scenario is gaze estimation for a person sitting at a desk. A front-facing RGB camera captures the face. The Vicon system provides camera/rig poses, a fixed table plane, and the real 3D gaze target point. There is no ground-truth face depth or true gaze direction; the reliable label is the target point on the table.

The recommended V1 direction is:

```text
eye image features
+ DECA face/global features
+ DECA pose/shape/expression/cam parameters
+ crop/bbox/camera intrinsic metadata
+ table/camera scene geometry
-> table-local gaze point (u, v)
-> 3D point on table
```

V1 should predict table-local `(u, v)`, not image coordinates and not direct unconstrained world `(X,Y,Z)`.

## V1 Model Architecture

Inputs:

```text
I_face:       [B, 3, 224, 224]
I_left_eye:   [B, 3, 36, 60]
I_right_eye:  [B, 3, 36, 60]
crop_cam_vec: [B, 36]
scene_vec:    [B, 25]  # updated from 22 after choosing full 3x3 rotation
```

Recommended first implementation:

```text
Face branch:
  DECA Encoder -> feature_deca_2048 [B,2048]
  MLP_deca: 2048 -> 512 -> 128
  f_deca: [B,128]

DECA parameter branch:
  shape [B,100] -> MLP -> [B,32]
  exp   [B,50]  -> MLP -> [B,16]
  pose  [B,3]   -> MLP -> [B,16]
  cam   [B,3]   -> MLP -> [B,8]
  concat [B,72] -> MLP -> f_deca_param [B,64]

Eye branch:
  Shared modified ResNet-18 / small CNN for both eyes.
  Input eye size is width 60, height 36, so PyTorch tensor is [B,3,36,60].
  Use conv1 3x3 stride 1 and remove maxpool.
  f_left:  [B,128]
  f_right: [B,128]
  concat(f_left, f_right, f_left - f_right, f_left * f_right) -> [B,512]
  MLP_eye_fusion: 512 -> 256 -> 256
  f_eye: [B,256]

Crop/camera branch:
  crop_cam_vec [B,36] -> MLP 36 -> 128 -> 64
  f_crop_cam: [B,64]

Scene branch:
  scene_vec [B,25] -> MLP 25 -> 128 -> 64
  f_scene: [B,64]

Fusion:
  concat(f_eye, f_deca, f_deca_param, f_crop_cam, f_scene)
  dimensions: 256 + 128 + 64 + 64 + 64 = 576
  MLP_fusion: 576 -> 512 -> 256
  f_all: [B,256]

Output:
  uv_head: 256 -> 128 -> 2
  uv_pred: [B,2]
```

Optional later module:

```text
DECA eye geometry branch:
  eye_geom_vec around [B,48] -> MLP -> f_face_geom [B,64]
  If added, fusion input becomes 640 dims.
```

V1 should not require `gaze_head -> g_C` or `origin_head -> O_eye_C`, because there is no reliable 3D eye-origin supervision. These can be V2 auxiliary/geometric branches after the direct `uv` model is stable.

## Important Model Decisions

- Do not treat the DECA 2048 feature as reliable depth by itself. Cropping the face to 224x224 removes much of the original scale cue.
- Real depth/space cues should come from bbox size/location, crop transforms, camera intrinsics, table geometry, and camera pose.
- First fusion method should be `concat + MLP`, not Cross Attention. Attention/FiLM/gated fusion can be tried after the baseline works.
- `f_left - f_right` is elementwise subtraction, capturing left/right difference.
- `f_left * f_right` is elementwise multiplication, capturing consistency/interaction.
- `uv_pred` represents table-local coordinates. It should be converted by:

```text
P_pred_W = O_table_W + u * e1_W + v * e2_W
```

## crop_cam_vec Definition

Original proposed size remains valid:

```text
crop_cam_vec: [B,36]
```

Order:

```text
face bbox normalized:       [xc/W, yc/H, w/W, h/H]       4
left eye bbox normalized:   [xc/W, yc/H, w/W, h/H]       4
right eye bbox normalized:  [xc/W, yc/H, w/W, h/H]       4
image size normalized:      [W/W_ref, H/H_ref]           2
camera intrinsics:          [fx/W, fy/H, cx/W, cy/H]     4
face crop affine 2x3:       [a11,a12,a13,a21,a22,a23]    6
left eye crop affine 2x3:                                      6
right eye crop affine 2x3:                                     6
Total: 36
```

Sources:

- `D:\GithubCode\CrossGaze-main\baseline\data_collection\dataset_dual_rigid_body_*\insightface_coordinates.csv`
- `detect_with_insightface.py`
- image/crop dirs: `insightface_img/`, `insightface_face/`, `insightface_eyes/left_eye/`, `insightface_eyes/right_eye/`

Field mapping:

```text
face_x, face_y, face_width, face_height
left_eye_x, left_eye_y, left_eye_width, left_eye_height
right_eye_x, right_eye_y, right_eye_width, right_eye_height
```

These are top-left plus width/height. Convert to center-normalized bbox:

```text
xc/W = (x + 0.5*w) / W
yc/H = (y + 0.5*h) / H
w/W  = w / W
h/H  = h / H
```

Known image/intrinsic constants:

```text
W = 1920
H = 1080
W_ref = 1920
H_ref = 1080
image size normalized = [1.0, 1.0]

fx = 1367.8584
fy = 1369.0087
cx = 957.9159
cy = 543.3381

[fx/W, fy/H, cx/W, cy/H] approx [0.7124, 1.2676, 0.4989, 0.5031]
```

Crop affine is not saved and must be reconstructed from bbox and target crop size. If using original image coordinates to crop coordinates:

```text
[[target_w / w, 0, -x * target_w / w],
 [0, target_h / h, -y * target_h / h]]
```

Face target size is `224x224`; eye target size is `60x36`.

## scene_vec Definition

Original plan was `[B,22]` with `R_6d`, but final decision in the old task was to use full `3x3` rotation. Therefore:

```text
scene_vec: [B,25]
```

Order:

```text
table normal in camera:      n_C         3
table plane distance:        d_C         1
table origin in camera:      O_table_C   3
table basis in camera:       e1_C        3
                             e2_C        3
camera rotation matrix:      R_CW        9
camera translation:          t_CW        3
Total: 25
```

Use camera-coordinate convention:

```text
X_C = R_CW @ X_W + t_CW
```

Plane equation convention:

```text
n_C dot X_C = d_C
d_C = n_C dot O_table_C
```

## Camera Pose / Hand-Eye Conventions

Fields in `data_log_*.csv`:

```text
cam_tx, cam_ty, cam_tz
cam_r11 ... cam_r33
target_tx, target_ty, target_tz
gaze_target_tx, gaze_target_ty, gaze_target_tz
tar_cam_tx, tar_cam_ty, tar_cam_tz
gaze_cam_tx, gaze_cam_ty, gaze_cam_tz
```

Important:

- `cam_tx/cam_r*` are the raw Vicon camera rigid-body pose and have not been multiplied by the hand-eye calibration matrix.
- `target_tx/tar_cam_tx` refer to the marker ball/object rigid body, not the real gaze target.
- Use `gaze_target_tx/y/z` as the real Vicon/world gaze target.
- `gaze_cam_tx/y/z` are intended to be the real gaze target in camera coordinates, but old code likely has a hand-eye translation unit bug.

Coordinate notation:

```text
W = Vicon/world coordinate system
R = camera rigid marker coordinate system
C = camera optical/camera coordinate system

R_WR = data_log cam_r11...cam_r33
t_WR = [cam_tx, cam_ty, cam_tz] in mm

R_RC = handeye[:3, :3]
t_RC = handeye[:3, 3] * 1000  # hand-eye translation was in meters

R_WC = R_WR @ R_RC
t_WC = R_WR @ t_RC + t_WR

R_CW = R_WC.T
t_CW = -R_CW @ t_WC
```

V1 labels should recompute `gaze_cam_*`:

```text
gaze_target_W = [gaze_target_tx, gaze_target_ty, gaze_target_tz]
gaze_cam_recomputed = R_CW @ gaze_target_W + t_CW
```

Then compare with old CSV `gaze_cam_tx/y/z` to quantify the hand-eye unit issue.

## Table Definition

The table is a plane perpendicular to the world/Vicon z-axis:

```text
z_table = mean(gaze_target_tz)
n_W = [0, 0, 1]
e1_W = [1, 0, 0]
e2_W = [0, 1, 0]
```

The table origin was updated during discussion. Do not use distant Vicon origin projection as final V1 origin. Use current camera optical center projected onto the table:

```text
C_W = t_WC
O_table_W = [C_W.x, C_W.y, z_table]
```

Then:

```text
n_C = R_CW @ n_W
O_table_C = R_CW @ O_table_W + t_CW
e1_C = R_CW @ e1_W
e2_C = R_CW @ e2_W
d_C = n_C dot O_table_C
```

This means `O_table_W` is frame-dependent. That is intentional because it keeps table-local coordinates numerically stable near the camera.

## Data Availability / Alignment

Use filename-based inner joins, not row order:

```text
data_log.image_filename == insightface_coordinates.image_name == pnp.image_name
```

Observed availability:

```text
dataset_1: data_log 100, InsightFace aligned 100, no PnP
dataset_2: data_log 368, InsightFace aligned 320, no PnP
dataset_3: data_log 506, InsightFace/PnP aligned 506
dataset_4: data_log 398, InsightFace/PnP aligned 398
dataset_5: data_log 229, InsightFace/PnP aligned 229
dataset_6: data_log 635, InsightFace/PnP CSV aligned 635, but missing insightface_* image dirs
dataset_7: data_log 855, InsightFace/PnP CSV aligned 855, but missing insightface_* image dirs
```

Other notes:

- Dataset 5 PnP rows may be one more than `data_log`; use filename join.
- Datasets 6 and 7 can support label/vector preparation but need image directories regenerated or restored for image training.
- Dataset 2 has incomplete InsightFace alignment: 320/368.

## Current Confidence / Risks

There was estimated 90-95% confidence that the data can be consolidated into one table/NPZ.

Main risks:

- Old `gaze_cam_*` may be wrong by the hand-eye translation unit mismatch.
- `scene_vec` must be treated as 25D now, not 22D.
- `d_C` sign convention must remain `n_C dot X_C = d_C`.
- Table origin is per-frame camera projection, not a fixed global table origin.
- `crop_affine` must be reconstructed consistently with the actual crop direction expected by the model.
- Datasets 6/7 lack image dirs; dataset 2 has missing InsightFace matches.

## Recommended Next Step

Write a data consolidation script that outputs a unified CSV/NPZ and a validation report:

- sample counts and inner-join counts per dataset
- missing image/crop/PnP checks
- rebuilt `crop_cam_vec` range checks
- rebuilt `scene_vec` shape checks, expecting 25D
- recomputed `gaze_cam_*` vs old CSV error statistics
- generated `uv_gt` under the per-frame table origin definition
