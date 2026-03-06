# Face ID

Face detection & recognition crate.

![img.png](.github/readme-img.png)

## Face detection models

This crate uses SCRDF face detection models. The following models are available:

> The naming convention for the ONNX models indicates the computational complexity (measured in FLOPs) and whether the
> model includes 5 facial keypoints predictions in addition to standard bounding boxes.

| Model Name            | Complexity        | Bounding Boxes | 5 Facial Keypoints |
|:----------------------|:------------------|:--------------:|:------------------:|
| **`500m.onnx`**       | 500 Million FLOPs |       ✅        |         ❌          |
| **`1g.onnx`**         | 1 Giga FLOPs      |       ✅        |         ❌          |
| **`2.5g_bnkps.onnx`** | 2.5 Giga FLOPs    |       ✅        |         ✅          |
| **`10g_bnkps.onnx`**  | 10 Giga FLOPs     |       ✅        |         ✅          |
| **`34g.onnx`**        | 34 Giga FLOPs     |       ✅        |         ❌          |
| **`34g_gnkps.onnx`**  | 34 Giga FLOPs     |       ✅        |         ❌          |

### Keypoints (`kps`) and Normalization Types (`bn` vs `gn`)

- **`kps`**: Denotes models that output 5 facial landmarks (keypoints) in addition to the standard bounding boxes.
- **`bnkps`**: Models trained using **Batch Normalization (BN)**. These often have lower false-positive rates and high
  recall on general datasets. However, they occasionally struggle with producing accurate landmarks for faces that are
  rotated past 90 degrees or are unusually large.
- **`gnkps`**: Models trained using **Group Normalization (GN)**. These variants (e.g., `34g_gnkps` or `10g_gnkps`) were
  explicitly developed to fix issues with very large faces that the `bnkps` models exhibited. While they improve
  landmark quality on large or rotated faces, they might have slightly lower general recall than `bnkps`.
