## 1. Preparation

- [x] 1.1 Ensure develop branch is clean (no uncommitted changes)
- [x] 1.2 Fetch latest from upstream (`git fetch upstream`)
- [x] 1.3 Review upstream changes for potential conflicts

## 2. Merge Upstream

- [x] 2.1 Create backup branch (`git branch develop-backup`)
- [x] 2.2 Merge upstream/master into develop
- [x] 2.3 Resolve any merge conflicts
- [x] 2.4 Run linting to verify merge didn't break anything

## 3. Verification

- [x] 3.1 Run quick sanity tests
- [x] 3.2 Verify openspec changes are still valid
- [x] 3.3 Check that our custom code still exists

## 4. Push

- [x] 4.1 Push updated develop to origin
- [x] 4.2 Delete backup branch if merge successful
