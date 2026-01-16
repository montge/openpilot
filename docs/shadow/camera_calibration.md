# Shadow Device Camera Calibration

This guide covers calibrating the OnePlus 6 camera for use as a shadow device alongside a production comma device.

## Overview

For accurate algorithm comparison, both devices must:
1. View the same scene from similar perspectives
2. Have known intrinsic camera parameters
3. Have synchronized timestamps

## Camera Specifications

### OnePlus 6 Cameras

| Camera | Sensor | Resolution | FOV | Notes |
|--------|--------|------------|-----|-------|
| Front | IMX371 | 16MP | ~80 | Working on postmarketOS |
| Rear Main | IMX519 | 16MP | ~75 | In development |
| Rear Secondary | IMX376 | 20MP | ~75 | Working on postmarketOS |

### comma three Road Camera

| Camera | Resolution | FOV | Notes |
|--------|------------|-----|-------|
| Road (narrow) | 1928x1208 | 40 | Main driving camera |
| Road (wide) | 1928x1208 | 120 | Peripheral vision |

## Calibration Process

### Step 1: Intrinsic Calibration

Determine the camera's intrinsic parameters (focal length, principal point, distortion).

#### Using OpenCV Checkerboard

1. Print a checkerboard pattern (9x6 internal corners recommended)

2. Capture 20+ images from various angles:
   ```python
   import cv2

   cap = cv2.VideoCapture(0)
   images = []

   while len(images) < 20:
       ret, frame = cap.read()
       cv2.imshow('Capture', frame)
       key = cv2.waitKey(1)
       if key == ord('c'):
           images.append(frame)
           print(f"Captured {len(images)}/20")
   ```

3. Calibrate using OpenCV:
   ```python
   import numpy as np
   import cv2
   import glob

   # Checkerboard dimensions
   CHECKERBOARD = (9, 6)

   # Prepare object points
   objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
   objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

   objpoints = []  # 3D points
   imgpoints = []  # 2D points

   for img in images:
       gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

       if ret:
           objpoints.append(objp)
           corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
           imgpoints.append(corners2)

   # Calibrate
   ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
       objpoints, imgpoints, gray.shape[::-1], None, None
   )

   print(f"Camera Matrix:\n{mtx}")
   print(f"Distortion Coefficients:\n{dist}")
   ```

4. Save calibration:
   ```python
   import json

   calibration = {
       "camera_matrix": mtx.tolist(),
       "distortion_coefficients": dist.tolist(),
       "image_size": list(gray.shape[::-1]),
       "reprojection_error": ret
   }

   with open("oneplus6_calibration.json", "w") as f:
       json.dump(calibration, f, indent=2)
   ```

### Step 2: Extrinsic Calibration (Mount Position)

Determine the camera's position relative to the vehicle.

#### Measuring Mount Position

1. **Height**: Distance from ground to camera center
2. **Lateral offset**: Distance from vehicle centerline
3. **Longitudinal offset**: Distance from front axle

```python
# Example extrinsic calibration
extrinsics = {
    "device_type": "oneplus6_shadow",
    "mount_position": {
        "x": 0.0,      # Lateral offset (m), positive = right
        "y": 1.2,      # Height (m)
        "z": 1.5,      # Longitudinal offset from front axle (m)
    },
    "mount_orientation": {
        "roll": 0.0,   # Rotation around forward axis (deg)
        "pitch": 0.0,  # Rotation around lateral axis (deg)
        "yaw": 0.0,    # Rotation around vertical axis (deg)
    }
}
```

### Step 3: Cross-Device Alignment

Ensure both cameras view the same scene similarly.

#### Physical Alignment

1. **Parallel Mounting**:
   - Mount shadow device parallel to production device
   - Minimize lateral offset (< 10cm if possible)
   - Match camera heights

2. **Angle Verification**:
   - Both cameras should point at same horizon
   - Use level to verify pitch alignment

#### Software Alignment

1. **Capture simultaneous frames**:
   ```python
   # Both devices capture same scene
   # Use GPS time or NTP for synchronization
   ```

2. **Compute homography**:
   ```python
   import cv2
   import numpy as np

   # Find matching features
   orb = cv2.ORB_create()
   kp1, des1 = orb.detectAndCompute(shadow_img, None)
   kp2, des2 = orb.detectAndCompute(prod_img, None)

   # Match features
   bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
   matches = bf.match(des1, des2)

   # Extract matched points
   pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
   pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

   # Compute homography
   H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

   print(f"Homography Matrix:\n{H}")
   ```

3. **Verify alignment quality**:
   ```python
   # Warp shadow image to production perspective
   warped = cv2.warpPerspective(shadow_img, H, (prod_img.shape[1], prod_img.shape[0]))

   # Compute difference
   diff = cv2.absdiff(warped, prod_img)
   alignment_error = np.mean(diff)

   print(f"Mean alignment error: {alignment_error}")
   ```

## Calibration Validation

### Checklist

- [ ] Intrinsic calibration reprojection error < 0.5 pixels
- [ ] Extrinsic measurements accurate to 1cm
- [ ] Cross-device homography residual < 5 pixels
- [ ] Lane lines align between devices
- [ ] Horizon line matches between devices

### Validation Script

```python
#!/usr/bin/env python3
"""Validate shadow device calibration."""

import json
import numpy as np
import cv2

def validate_intrinsics(calibration_file: str) -> bool:
    with open(calibration_file) as f:
        cal = json.load(f)

    error = cal.get("reprojection_error", float("inf"))
    if error > 0.5:
        print(f"WARNING: Reprojection error {error:.3f} > 0.5")
        return False

    print(f"Intrinsic calibration OK (error={error:.3f})")
    return True

def validate_alignment(shadow_img, prod_img, H) -> bool:
    warped = cv2.warpPerspective(shadow_img, H,
                                  (prod_img.shape[1], prod_img.shape[0]))

    # Convert to grayscale for comparison
    gray1 = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(prod_img, cv2.COLOR_BGR2GRAY)

    # Compute structural similarity
    from skimage.metrics import structural_similarity as ssim
    score = ssim(gray1, gray2)

    if score < 0.8:
        print(f"WARNING: SSIM {score:.3f} < 0.8")
        return False

    print(f"Alignment OK (SSIM={score:.3f})")
    return True

if __name__ == "__main__":
    validate_intrinsics("oneplus6_calibration.json")
```

## Integration with openpilot

### Calibration File Location

Store calibration in the standard location:
```
/data/params/d/CalibrationParams
```

### Format

openpilot expects calibration in a specific format:
```python
from openpilot.common.params import Params

params = Params()

# Read existing calibration
cal = params.get("CalibrationParams")

# For shadow device, store separately
params.put("ShadowCalibrationParams", json.dumps({
    "intrinsics": {...},
    "extrinsics": {...},
    "device_type": "oneplus6"
}))
```

### Using Calibration in Shadow Mode

```python
from openpilot.system.hardware.shadow_mode import is_shadow_mode

if is_shadow_mode():
    # Load shadow device calibration
    cal = load_shadow_calibration()
else:
    # Load production calibration
    cal = load_production_calibration()
```

## Troubleshooting

### High Reprojection Error

- Ensure checkerboard is flat
- Use more calibration images
- Check for motion blur
- Verify checkerboard dimensions

### Poor Cross-Device Alignment

- Verify cameras are physically parallel
- Check for lens distortion differences
- Ensure same scene is captured
- Verify timestamp synchronization

### Lane Lines Don't Align

- Check camera pitch angle
- Verify mounting height is correct
- Recalibrate extrinsics

## Next Steps

1. Complete intrinsic calibration
2. Mount device and measure extrinsics
3. Capture validation dataset
4. Run comparison tests

See [comparison testing](../../tools/shadow/README.md) for next steps.
