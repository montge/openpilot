## 1. Setup Scripts

- [x] 1.1 Create `tools/shadow/setup/` directory structure
- [x] 1.2 Create `termux-setup.sh` script
  - [x] 1.2.1 Update Termux packages
  - [x] 1.2.2 Install wget, curl, git, python, proot-distro
  - [x] 1.2.3 Install Ubuntu via proot-distro
- [x] 1.3 Create `ubuntu-setup.sh` script
  - [x] 1.3.1 Update Ubuntu packages
  - [x] 1.3.2 Install build-essential, git, git-lfs, python3, clang, cmake
  - [x] 1.3.3 Configure git-lfs
- [x] 1.4 Create `clone-openpilot.sh` script
  - [x] 1.4.1 Clone openpilot repository
  - [x] 1.4.2 Initialize submodules
  - [x] 1.4.3 Create Python virtual environment
  - [x] 1.4.4 Install base Python dependencies
- [x] 1.5 Create `setup-ssh.sh` script
  - [x] 1.5.1 Install OpenSSH in Termux
  - [x] 1.5.2 Configure SSH server on port 8022
  - [x] 1.5.3 Display connection instructions

## 2. Documentation

- [x] 2.1 Create `tools/shadow/setup/README.md`
  - [x] 2.1.1 Document prerequisites (OnePlus 6, unlocked bootloader)
  - [x] 2.1.2 Document LineageOS flashing procedure
  - [x] 2.1.3 Document Termux installation from F-Droid
  - [x] 2.1.4 Document script execution order
  - [x] 2.1.5 Document SSH remote development setup
  - [x] 2.1.6 Document shadow mode verification
- [x] 2.2 Create `tools/shadow/setup/FLASHING.md`
  - [x] 2.2.1 Document firmware version requirements (Android 11)
  - [x] 2.2.2 Document required downloads (boot.img, dtbo.img, lineage.zip)
  - [x] 2.2.3 Document fastboot flashing commands
  - [x] 2.2.4 Document recovery mode sideload procedure
- [x] 2.3 Create `tools/shadow/setup/TROUBLESHOOTING.md`
  - [x] 2.3.1 Document common proot issues
  - [x] 2.3.2 Document SSH connection issues
  - [x] 2.3.3 Document device detection issues

## 3. Shadow Mode Detection Update

- [x] 3.1 Add `_get_android_device()` function to `shadow_mode.py`
- [x] 3.2 Update `is_oneplus6()` to use getprop fallback
- [x] 3.3 Update `clear_shadow_mode_cache()` to clear new cache
- [ ] 3.4 Add unit tests for Android getprop detection path
- [ ] 3.5 Update shadow mode README with proot notes

## 4. Camera Integration

- [x] 4.1 Research camera access methods
  - [x] 4.1.1 Test termux-api camera access (works for snapshots)
  - [x] 4.1.2 Test V4L2 access (not available without root)
  - [x] 4.1.3 Test proot camera access via termux-api PATH export
- [x] 4.2 Document camera findings
  - [x] 4.2.1 termux-camera-photo works for single frames
  - [x] 4.2.2 Continuous streaming requires alternative approach
  - [x] 4.2.3 IP Webcam / RTSP server approach for video streaming
- [x] 4.3 Create camera bridge prototype (`camera_bridge.py`)
  - [x] 4.3.1 HTTP MJPEG capture from IP Webcam
  - [x] 4.3.2 BGR to NV12 conversion for openpilot
  - [x] 4.3.3 VisionIPC publishing integration
- [x] 4.4 Test camera streaming end-to-end
  - [x] 4.4.1 Install compatible camera streaming app (termux-api server)
  - [x] 4.4.2 Test frame capture from stream (3000x4000 captured)
  - [x] 4.4.3 Verify NV12 conversion accuracy (18MB output correct)
- [x] 4.5 Document camera setup procedure
  - [x] 4.5.1 Create CAMERA.md with streaming options
  - [x] 4.5.2 Document latency and performance characteristics
  - [x] 4.5.3 Document limitations for shadow mode use

## 5. Validation

- [x] 5.1 Test on fresh LineageOS 22.2 install
- [x] 5.2 Verify shadow mode detection works via SSH
- [x] 5.3 Document tested LineageOS version (22.2)
- [ ] 5.4 Run full openpilot pipeline in shadow mode
- [x] 5.5 Measure camera latency and frame rate (0.4 FPS termux-api, 15-30 FPS IP Webcam)

## 6. VisionIPC Integration

- [x] 6.1 Build msgq module in proot environment
  - [x] Install system deps: libzmq3-dev, ocl-icd-opencl-dev, opencl-headers
  - [x] Install Python deps: Cython, scons, setuptools
  - [x] Download catch2 headers for tests
  - [x] Run scons in msgq_repo directory
- [x] 6.2 Test VisionIPC server creation
- [x] 6.3 Run camera_bridge.py with VisionIPC publishing
- [x] 6.4 Test VisionIPC server/client communication
  - [x] Server creates buffers successfully
  - [x] Client connects and receives stream info
- [x] 6.5 Document build process for msgq in proot

## 7. OpenCL / GPU Access Investigation

- [x] 7.1 Check OpenCL library availability in proot
  - [x] libOpenCL.so.1 exists in /usr/lib/aarch64-linux-gnu/
  - [x] No ICD vendors configured (/etc/OpenCL/vendors/ empty)
- [x] 7.2 Check Android GPU driver availability
  - [x] Adreno OpenCL driver at /vendor/lib64/libOpenCL.so
  - [x] GPU model: Adreno630v2
- [x] 7.3 Investigate device access restrictions
  - [x] /dev/kgsl* not exposed (kernel device nodes)
  - [x] /dev/dri/ permission denied
  - [x] Root required for GPU device access
- [x] 7.4 Document findings and options
  - [x] Updated design.md with Decision 4
  - [x] OpenCL blocked without root (hard limitation)

## 8. Full modeld Integration (BLOCKED - requires root or alternative)

**Status**: Blocked by OpenCL/GPU access. Options:
1. Root the device to enable /dev/kgsl access
2. Implement remote inference server (desktop GPU)
3. Accept partial pipeline (VisionIPC works, modeld doesn't)

- [ ] 8.1 (Option A) Root device and test OpenCL
  - [ ] Flash Magisk or similar root solution
  - [ ] Verify /dev/kgsl* accessible
  - [ ] Test OpenCL with clinfo
  - [ ] Run modeld with GPU acceleration
- [ ] 8.2 (Option B) Remote inference server
  - [ ] Create frame streaming protocol
  - [ ] Implement server-side modeld wrapper
  - [ ] Send inference results back to device
- [ ] 8.3 (Option C) CPU-only inference (experimental)
  - [ ] Investigate tinygrad CPU backend
  - [ ] Benchmark inference speed
  - [ ] Evaluate viability for shadow mode
