## 1. Setup Scripts

- [x] 1.1 Create `tools/shadow/setup/` directory structure
- [ ] 1.2 Create `termux-setup.sh` script
  - [ ] 1.2.1 Update Termux packages
  - [ ] 1.2.2 Install wget, curl, git, python, proot-distro
  - [ ] 1.2.3 Install Ubuntu via proot-distro
- [ ] 1.3 Create `ubuntu-setup.sh` script
  - [ ] 1.3.1 Update Ubuntu packages
  - [ ] 1.3.2 Install build-essential, git, git-lfs, python3, clang, cmake
  - [ ] 1.3.3 Configure git-lfs
- [ ] 1.4 Create `clone-openpilot.sh` script
  - [ ] 1.4.1 Clone openpilot repository
  - [ ] 1.4.2 Initialize submodules
  - [ ] 1.4.3 Create Python virtual environment
  - [ ] 1.4.4 Install base Python dependencies
- [ ] 1.5 Create `setup-ssh.sh` script
  - [ ] 1.5.1 Install OpenSSH in Termux
  - [ ] 1.5.2 Configure SSH server on port 8022
  - [ ] 1.5.3 Display connection instructions

## 2. Documentation

- [ ] 2.1 Create `tools/shadow/setup/README.md`
  - [ ] 2.1.1 Document prerequisites (OnePlus 6, unlocked bootloader)
  - [ ] 2.1.2 Document LineageOS flashing procedure
  - [ ] 2.1.3 Document Termux installation from F-Droid
  - [ ] 2.1.4 Document script execution order
  - [ ] 2.1.5 Document SSH remote development setup
  - [ ] 2.1.6 Document shadow mode verification
- [ ] 2.2 Create `tools/shadow/setup/FLASHING.md`
  - [ ] 2.2.1 Document firmware version requirements (Android 11)
  - [ ] 2.2.2 Document required downloads (boot.img, dtbo.img, lineage.zip)
  - [ ] 2.2.3 Document fastboot flashing commands
  - [ ] 2.2.4 Document recovery mode sideload procedure
- [ ] 2.3 Create `tools/shadow/setup/TROUBLESHOOTING.md`
  - [ ] 2.3.1 Document common proot issues
  - [ ] 2.3.2 Document SSH connection issues
  - [ ] 2.3.3 Document device detection issues

## 3. Shadow Mode Detection Update

- [x] 3.1 Add `_get_android_device()` function to `shadow_mode.py`
- [x] 3.2 Update `is_oneplus6()` to use getprop fallback
- [x] 3.3 Update `clear_shadow_mode_cache()` to clear new cache
- [ ] 3.4 Add unit tests for Android getprop detection path
- [ ] 3.5 Update shadow mode README with proot notes

## 4. Validation

- [ ] 4.1 Test scripts on fresh LineageOS install
- [ ] 4.2 Verify shadow mode detection works via SSH
- [ ] 4.3 Document tested LineageOS version (22.2)
- [ ] 4.4 Run openspec validate --strict
