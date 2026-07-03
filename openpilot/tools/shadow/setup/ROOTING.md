# Rooting LineageOS for OpenCL Access

This guide covers rooting the OnePlus 6 with Magisk to enable GPU/OpenCL access for modeld.

## Why Root?

Without root, the proot environment cannot access GPU device nodes:
- `/dev/kgsl*` - Adreno GPU kernel interface
- `/dev/dri/` - Direct Rendering Infrastructure

Rooting allows Termux to access these devices, enabling OpenCL for modeld inference.

## Prerequisites

- OnePlus 6 with LineageOS already installed (see [FLASHING.md](FLASHING.md))
- The `boot.img` from your LineageOS version
- Computer with `adb` and `fastboot`
- USB cable

## Step 1: Download Magisk

On the phone, download Magisk from the official GitHub:

https://github.com/topjohnwu/Magisk/releases

Download the latest `Magisk-v*.apk` (e.g., `Magisk-v28.1.apk`).

Install the APK:
1. Open Files app, navigate to Downloads
2. Tap the APK file
3. Allow installation from unknown sources if prompted
4. Install Magisk

## Step 2: Get Your Current boot.img

You need the exact boot.img matching your LineageOS version.

**Option A: Download from LineageOS**
```bash
# On computer, check your LineageOS version first
adb shell getprop ro.lineage.version
# Example output: 22.1-20250115-NIGHTLY-enchilada

# Download matching boot.img from:
# https://download.lineageos.org/devices/enchilada/builds
```

**Option B: Extract from device (if you have it)**
If you saved the boot.img from initial flashing, use that.

## Step 3: Transfer boot.img to Phone

```bash
# On computer
adb push boot.img /sdcard/Download/boot.img
```

## Step 4: Patch boot.img with Magisk

On the phone:
1. Open the **Magisk** app
2. Tap **Install** next to "Magisk" at the top
3. Select **Select and Patch a File**
4. Navigate to Downloads and select `boot.img`
5. Tap **LET'S GO**

Magisk will create a patched file like:
`/sdcard/Download/magisk_patched-XXXXX_XXXXX.img`

## Step 5: Transfer Patched boot.img to Computer

```bash
# On computer
adb pull /sdcard/Download/magisk_patched-*.img ./magisk_patched.img
```

## Step 6: Flash Patched boot.img

```bash
# Reboot to bootloader
adb reboot bootloader

# Wait for fastboot mode, then flash
fastboot flash boot magisk_patched.img

# Reboot
fastboot reboot
```

## Step 7: Verify Root

After reboot, open the Magisk app. You should see:
- **Magisk**: Installed (version number)
- **App**: Up to date

Test root in Termux:
```bash
su
# Should show root shell prompt (#)
whoami
# Should show: root
```

## Step 8: Grant Termux Root Access

When Termux requests root:
1. A Magisk popup will appear
2. Tap **Grant**
3. Optionally check "Remember choice"

Or pre-grant in Magisk:
1. Open Magisk app
2. Tap **Superuser** at bottom
3. Find Termux in the list
4. Toggle to allow

## Step 9: Test OpenCL Access

In Termux (not proot):
```bash
# Check if kgsl devices are visible
ls -la /dev/kgsl*
# Should show: /dev/kgsl-3d0

# In proot Ubuntu
proot-distro login ubuntu
apt install -y clinfo
clinfo
# Should now show Adreno GPU
```

## Step 10: Configure proot for GPU Access

The proot environment needs explicit device bindings:
```bash
# In Termux (not proot)
proot-distro login ubuntu --bind /dev/kgsl-3d0:/dev/kgsl-3d0

# Or add to login alias in ~/.bashrc:
alias ubuntu='proot-distro login ubuntu --bind /dev/kgsl-3d0:/dev/kgsl-3d0'
```

## Verification Commands

```bash
# Check GPU device access (in Termux with root)
su -c "ls -la /dev/kgsl*"

# Check OpenCL (MUST run as root due to linker namespace)
su -c "clinfo | head -20"
# Should show: QUALCOMM Snapdragon(TM) / Adreno(TM) 630
```

## Important: Android Linker Namespace Limitation

Even with root, Android's linker namespace isolation prevents regular user processes from loading `/vendor/lib64/libOpenCL.so`. This means:

- **OpenCL only works when running as root** (`su -c "command"`)
- **proot Ubuntu cannot use OpenCL** - the Android library uses Bionic libc, not glibc

**What works after rooting:**
| Component | Status | Notes |
|-----------|--------|-------|
| GPU device access | Works | `/dev/kgsl-3d0` accessible |
| OpenCL (Termux + root) | Works | `su -c clinfo` shows Adreno 630 |
| OpenCL (proot) | Blocked | glibc/Bionic incompatibility |
| VisionIPC | Works | Frame publishing/consuming |

**Implication for modeld:**
modeld requires OpenCL for frame transforms. Options:
1. Build modeld for Termux directly (uses Bionic, can access OpenCL as root)
2. Use remote inference server (send frames to desktop GPU)
3. Run relevant components through `su -c` wrapper

## Troubleshooting

### Magisk app shows "N/A" for installation
- The patched boot.img wasn't flashed correctly
- Re-flash: `fastboot flash boot magisk_patched.img`

### "su: not found" in Termux
- Magisk may need reinstallation
- Or grant Termux superuser access in Magisk app

### proot still can't see /dev/kgsl*
- Make sure to use `--bind` option when starting proot
- Check device exists in Termux first: `ls /dev/kgsl*`

### "Permission denied" on /dev/kgsl-3d0
- SELinux may be blocking
- Try: `su -c setenforce 0` (temporary, resets on reboot)
- Or use Magisk's "Enforce DenyList" settings

### OpenCL shows 0 platforms in proot
May need to bind additional paths:
```bash
proot-distro login ubuntu \
  --bind /dev/kgsl-3d0:/dev/kgsl-3d0 \
  --bind /vendor:/vendor \
  --bind /system:/system
```

## Security Considerations

Rooting has security implications:
- Apps can request root access (Magisk prompts for approval)
- Some apps detect root and refuse to run (banking apps)
- SafetyNet/Play Integrity will fail (no Google services on LineageOS anyway)

For shadow device development use, these tradeoffs are acceptable.

## OTA Updates with Magisk

When updating LineageOS:
1. Download new LineageOS zip
2. Download matching new boot.img
3. Patch new boot.img with Magisk
4. Apply OTA update normally
5. Before rebooting, flash patched boot.img to inactive slot

See: https://topjohnwu.github.io/Magisk/ota.html

## Next Steps

After rooting:
1. Test OpenCL access with clinfo
2. Build/test modeld with GPU support
3. Run full openpilot pipeline in shadow mode
