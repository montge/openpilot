## ADDED Requirements

### Requirement: OnePlus 6 LineageOS Setup Scripts

The system SHALL provide automated setup scripts for configuring OnePlus 6 as a shadow device running LineageOS with Termux and proot-distro Ubuntu.

#### Scenario: Fresh device setup
- **WHEN** a user has a OnePlus 6 with LineageOS installed
- **AND** runs the Termux setup script
- **THEN** proot-distro with Ubuntu is installed and configured

#### Scenario: Ubuntu environment setup
- **WHEN** a user runs the Ubuntu setup script inside proot
- **THEN** all openpilot build dependencies are installed

#### Scenario: Openpilot clone and configure
- **WHEN** a user runs the clone script
- **THEN** openpilot is cloned with submodules and Python venv configured

### Requirement: Android Getprop Device Detection

The shadow mode detection system SHALL support Android `getprop` as a fallback method for device identification when running in proot/chroot environments where `/sys/firmware/devicetree/base/model` is inaccessible.

#### Scenario: Device detection in proot environment
- **WHEN** openpilot runs on OnePlus 6 inside proot-distro
- **AND** the device tree file is not accessible
- **THEN** the system detects OnePlus 6 via `getprop ro.product.device`
- **AND** shadow mode is automatically enabled

#### Scenario: Native device detection preserved
- **WHEN** openpilot runs natively on OnePlus 6 (not in proot)
- **THEN** the system detects OnePlus 6 via device tree first
- **AND** falls back to getprop only if device tree unavailable

### Requirement: SSH Remote Development Support

The shadow device setup SHALL include SSH server configuration for remote development access.

#### Scenario: SSH server setup
- **WHEN** a user runs the SSH setup script in Termux
- **THEN** OpenSSH server starts on port 8022
- **AND** the user can connect via `ssh user@device-ip -p 8022`

#### Scenario: SSH key authentication
- **WHEN** a user adds their public key to authorized_keys
- **THEN** passwordless SSH authentication is enabled
