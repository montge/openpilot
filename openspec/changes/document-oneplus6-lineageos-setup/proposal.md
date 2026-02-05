# Change: Document OnePlus 6 LineageOS Setup for Shadow Mode

## Why

The existing shadow device proposal (add-shadow-device-harness) assumed AGNOS would work on OnePlus 6, but AGNOS is built for comma 3/3X custom hardware. Through experimentation, we discovered that **LineageOS + Termux + proot-distro** is a viable approach for running openpilot on OnePlus 6 as a shadow device.

This documentation and tooling is needed because:
1. **No existing guide**: The OnePlus 6 setup process is undocumented
2. **Non-obvious approach**: Using proot to run Ubuntu inside Termux on LineageOS is not intuitive
3. **Shadow mode fix**: The device detection code needed modification to work in proot environments
4. **Reproducibility**: Scripts ensure consistent setup across devices

## What Changes

- **Add Setup Scripts** (`tools/shadow/setup/`):
  - `flash-lineageos.md` - Step-by-step LineageOS flashing guide
  - `termux-setup.sh` - Termux package installation and proot-distro setup
  - `ubuntu-setup.sh` - Ubuntu dependencies for openpilot
  - `clone-openpilot.sh` - Clone and configure openpilot fork
  - `setup-ssh.sh` - SSH server for remote development

- **Update Shadow Mode Detection** (already committed):
  - Added `_get_android_device()` fallback using `getprop ro.product.device`
  - Enables detection in proot/Termux where `/sys/firmware/devicetree/base/model` is inaccessible

- **Add Setup Documentation** (`tools/shadow/setup/README.md`):
  - Complete setup guide from stock Android to working shadow device
  - Troubleshooting common issues
  - SSH remote development workflow

## Impact

- Affected specs: `shadow-device` (adds setup requirements)
- Affected code:
  - `system/hardware/shadow_mode.py` - Android getprop fallback (already merged)
  - `tools/shadow/setup/` - New directory with scripts and docs
- Dependencies: LineageOS 22+, Termux from F-Droid, proot-distro
- **SAFETY**: No changes to safety-critical code; shadow mode already prevents actuation
