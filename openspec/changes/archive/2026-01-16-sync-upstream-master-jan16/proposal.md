# Change: Sync Fork with Upstream Master (January 2026)

## Why
The fork is currently 3 commits behind upstream/master. These commits include bug fixes and improvements that should be incorporated to maintain compatibility with the upstream project and benefit from community fixes.

## What Changes
- Merge 3 upstream commits:
  - `7f8dbf24e` Cabana: fix for internal source (#36998)
  - `5e4b88201` Toyota: whitelist hybrids for standstill resume behavior (#36996)
  - `1252188b4` sensord: tighten temperature threshold (#36994)
- Verify all tests pass after merge
- Verify build succeeds
- Review formal verification status
- Ensure fork-specific changes (FAIR algorithms, DGX support, etc.) still function

## Impact
- Affected specs: development-workflow
- Affected code: tools/cabana/, selfdrive/car/toyota/, system/sensord/
- Risk: Low - these are bug fixes, not architectural changes
