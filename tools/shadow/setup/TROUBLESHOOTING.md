# Troubleshooting Guide

Common issues and solutions for OnePlus 6 shadow device setup.

## Termux Issues

### "pkg: command not found"
You may have the Play Store version of Termux which is outdated and broken.

**Solution:** Uninstall Termux, install from F-Droid:
```bash
adb uninstall com.termux
# Download from https://f-droid.org/packages/com.termux/
adb install termux_*.apk
```

### "Unable to resolve host" during pkg update
Network issue in Termux.

**Solution:**
```bash
# Try changing repository
termux-change-repo
# Select a mirror closer to you
```

### Storage permission denied
`termux-setup-storage` didn't complete properly.

**Solution:**
1. Go to Android Settings → Apps → Termux → Permissions
2. Enable Storage permission manually
3. Restart Termux

## Proot Issues

### "proot warning: can't sanitize binding /proc/self/fd/X"
This is **normal and harmless**. Proot can't access certain /proc paths because it's not true root.

**No action needed** - operations still complete successfully.

### "proot-distro: command not found"
Proot-distro package not installed.

**Solution:**
```bash
pkg install proot-distro
```

### Ubuntu proot very slow
Proot has overhead compared to native execution.

**Tips:**
- Use SSH from computer instead of on-device terminal
- Run CPU-intensive tasks during cooler times
- Ensure phone isn't thermal throttling

### "CANNOT LINK EXECUTABLE" in proot
Architecture mismatch or broken installation.

**Solution:**
```bash
# Remove and reinstall Ubuntu
proot-distro remove ubuntu
proot-distro install ubuntu
```

## SSH Issues

### Connection refused
SSH server not running.

**Solution in Termux:**
```bash
# Check if sshd is running
pgrep sshd

# If not, start it
sshd

# If it fails, check for port conflict
pkill sshd
sshd -d  # Debug mode shows errors
```

### Permission denied (publickey,password)
Wrong username or password.

**Solution:**
- Username in Termux is like `u0_a191` (check with `whoami`)
- Reset password: `passwd`
- For key auth: ensure `~/.ssh/authorized_keys` has correct permissions:
  ```bash
  chmod 700 ~/.ssh
  chmod 600 ~/.ssh/authorized_keys
  ```

### Connection times out
Phone IP may have changed (DHCP lease renewal).

**Solution:**
```bash
# On phone in Termux, get current IP:
ifconfig | grep -A1 wlan0 | grep inet
```

### SSH works but proot commands fail
Need to use proper quoting for nested commands.

**Solution:**
```bash
# Correct:
ssh user@ip -p 8022 "proot-distro login ubuntu -- bash -c 'echo hello'"

# Wrong (quote issues):
ssh user@ip -p 8022 proot-distro login ubuntu -- echo hello
```

## Shadow Mode Issues

### "OnePlus 6 detected: False"
Device not being recognized.

**Debug steps:**
```bash
# Check what getprop returns (in Ubuntu proot):
getprop ro.product.device
# Should return: OnePlus6

# Check device tree (likely fails in proot, that's ok):
cat /sys/firmware/devicetree/base/model
```

**If getprop doesn't work in proot:**
The host Android's getprop should be accessible. Try:
```bash
/system/bin/getprop ro.product.device
```

### "Shadow mode active: False"
Shadow mode requires either OnePlus 6 detection OR environment variable.

**Solution - force shadow mode:**
```bash
export SHADOW_MODE=1
python3 -c "from openpilot.system.hardware.shadow_mode import is_shadow_mode; print(is_shadow_mode())"
```

### Import errors when testing shadow mode
Missing Python dependencies.

**Solution:**
```bash
cd ~/openpilot
source .venv/bin/activate
pip install pycapnp numpy
```

## Build Issues

### "error: unable to execute clang"
Clang not installed in Ubuntu proot.

**Solution:**
```bash
apt install -y clang
```

### Git LFS files not downloaded
LFS not initialized or network issues.

**Solution:**
```bash
git lfs install
git lfs pull
```

### "No space left on device"
Phone storage full.

**Solution:**
- Clear LineageOS cache: Settings → Storage → Free up space
- Remove unused apps
- Use shallow clone: `git clone --depth 1`

## Performance Issues

### Phone getting very hot
CPU-intensive operations cause thermal throttling.

**Tips:**
- Remove phone case for better cooling
- Run heavy tasks in air-conditioned environment
- Use a small fan pointed at the phone
- Take breaks between intensive operations

### Operations randomly killed
Android OOM killer terminating Termux.

**Solution:**
1. Acquire Termux wakelock: `termux-wake-lock`
2. In Termux notification, tap "Acquire wakelock"
3. Disable battery optimization for Termux in Android settings

## Getting Help

If issues persist:

1. Check Termux wiki: https://wiki.termux.com/
2. Search Termux issues: https://github.com/termux/termux-app/issues
3. LineageOS wiki for OnePlus 6: https://wiki.lineageos.org/devices/enchilada/
