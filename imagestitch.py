import numpy as np
import cv2
import glob
import imutils

# Corrected path with raw string
image_paths = glob.glob(r'C:\MyData\Deepali\MyPrograms\PythonWork\images\*.jpg')

images = []

# Loading images
for image_path in image_paths:
    img = cv2.imread(image_path)
    if img is not None:
        # Resize image to make processing faster
        img = cv2.resize(img, (500, 500), fx=0.01, fy=0.01)
        images.append(img)
    else:
        print(f"Error loading image {image_path}")

# Convert images to grayscale for SIFT feature detection
gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect and compute key points and descriptors
keypoints_and_descriptors = [sift.detectAndCompute(img, None) for img in gray_images]

# Draw keypoints for visualization (optional)
for i, (kp, des) in enumerate(keypoints_and_descriptors):
    img_with_kp = cv2.drawKeypoints(images[i], kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv2.imshow(f"Keypoints {i+1}", img_with_kp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Prepare key points and descriptors for stitching
keypoints = [kp for kp, des in keypoints_and_descriptors]
descriptors = [des for kp, des in keypoints_and_descriptors]

# Using OpenCV Stitcher with SIFT
try:
    image_stitcher = cv2.Stitcher.create()  # For OpenCV 3.4+
except AttributeError:
    image_stitcher = cv2.Stitcher_create()  # For older OpenCV versions

# Stitch the images
error, stitched_img = image_stitcher.stitch(images)

stitched_img = cv2.copyMakeBorder(stitched_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (10,10,10))
for width in (400, 300, 200, 100):
    resized = imutils.resize(img, width = width)

# Handle stitching errors
if error == cv2.Stitcher_OK:
    cv2.imwrite("stitchedOutput.jpg", stitched_img)
    cv2.imshow("Stitched Img", stitched_img)
    cv2.waitKey(0)
elif error == cv2.Stitcher_ERR_NEED_MORE_IMGS:
    print("Not enough key points")
elif error == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
    print("Insufficient features")
elif error == cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
    print("Failed estimation of camera parameters")
else:
    print("Error stitching images")
