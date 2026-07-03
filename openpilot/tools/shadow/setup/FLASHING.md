# LineageOS Flashing Guide for OnePlus 6

Step-by-step guide to flash LineageOS on OnePlus 6 for shadow device use.

## Prerequisites

- OnePlus 6 with **unlocked bootloader**
- Computer with `adb` and `fastboot` installed
- USB cable (data-capable)
- ~2GB free space for downloads

## Step 1: Verify Firmware Version

LineageOS requires **Android 11 firmware** (OxygenOS 11.x).

On the phone: Settings → About Phone → Android version

Should show: Android 11 (or OxygenOS 11.x.x.x)

If on older version, update via OnePlus updater first.

## Step 2: Download Required Files

From https://download.lineageos.org/devices/enchilada/builds:

| File | Size | Purpose |
|------|------|---------|
| `lineage-22.x-*-nightly-enchilada-signed.zip` | ~1GB | The OS |
| `boot.img` | ~64MB | Lineage Recovery |
| `dtbo.img` | ~8MB | Device tree overlay |

Save all to your Downloads folder.

## Step 3: Enable USB Debugging

On the phone:
1. Settings → About Phone
2. Tap "Build Number" 7 times to enable Developer Options
3. Settings → System → Developer Options
4. Enable "USB Debugging"
5. Connect USB cable, authorize computer when prompted

Verify on computer:
```bash
adb devices
# Should show your device ID with "device" status
```

## Step 4: Boot to Fastboot

```bash
adb reboot bootloader
```

Or: Power off, then hold **Volume Up + Power** until fastboot screen appears.

Verify connection:
```bash
fastboot devices
# Should show your device ID
```

## Step 5: Flash Partitions

```bash
cd ~/Downloads  # or wherever you saved the files

# Flash device tree overlay
fastboot flash dtbo dtbo.img

# Reboot to bootloader (important!)
fastboot reboot bootloader

# Wait for device to reappear, then flash recovery
fastboot flash boot boot.img
```

## Step 6: Boot into Recovery

Use **Volume buttons** to navigate to "Recovery Mode", press **Power** to select.

You'll see the Lineage Recovery menu.

## Step 7: Factory Reset

In Lineage Recovery:
1. Select **Factory Reset**
2. Select **Format data/factory reset**
3. Confirm the reset
4. Return to main menu (back arrow)

## Step 8: Sideload LineageOS

In Lineage Recovery:
1. Select **Apply update**
2. Select **Apply from ADB**
3. Screen will show "waiting for sideload"

On computer:
```bash
adb sideload lineage-22.x-*-nightly-enchilada-signed.zip
```

Wait for transfer and installation to complete (~5-10 minutes).

## Step 9: Reboot

In Lineage Recovery:
1. Tap back arrow to return to main menu
2. Select **Reboot system now**

First boot takes longer than normal (~5-15 minutes). This is expected.

## Step 10: Initial Setup

Complete the LineageOS setup wizard:
- Skip Google account (or add if you want Play Store)
- Connect to WiFi
- Set up fingerprint/PIN (optional)

## Verification

After setup, verify in Settings → About Phone:
- Android version: 15 (LineageOS 22.x)
- Device: OnePlus 6

## Next Steps

Continue with the main [README.md](README.md) to:
1. Install Termux from F-Droid
2. Run setup scripts
3. Configure SSH access

## Troubleshooting

### "FAILED (remote: 'Partition flashing is not allowed')"
- Bootloader may not be unlocked
- Run: `fastboot oem unlock` (erases all data!)

### "FAILED (remote: 'Command not allowed')"
- Device might be in wrong slot
- Try: `fastboot set_active a` then retry

### Phone stuck in boot loop
- Boot back to recovery (Vol Down + Power)
- Try factory reset again
- Re-sideload LineageOS

### "adb sideload" stuck at 47%
- This is normal behavior; wait for completion
- Progress display is misleading during verification phase

### Recovery shows "No command"
- Press and hold Power, tap Volume Up once, release both
- Recovery menu should appear
