# TODO

* Face embedding
* rwlock zodat detector niet mut hoeft?
* batch processing optie
* face recog
    * face align
        * Key Implementation Challenge: Similarity Transform
          The hardest part of the ArcFace pipeline is the face alignment.
          skimage.transform.SimilarityTransform.estimate() solves for
          (scale, rotation, tx, ty)
          given point correspondences. In Rust, this can be done:

          Use the Umeyama algorithm (direct closed-form solution for similarity transforms between point sets — exactly
          what skimage uses internally).

          The Umeyama algorithm is the correct approach: given source points src (5 detected keypoints) and destination
          points dst (canonical arcface_dst), compute the 2×3 affine matrix that minimizes the sum of squared distances.
          It's a ~30-line algorithm involving means, covariances, and SVD decomposition.

          Then use the resulting 2×3 matrix to warpAffine the original image → 112×112 crop using bilinear
          interpolation.
    * face recog model: arcface -> public-data/insightface / buffalo_l
* age / gender estimation
* get padded cropped faces