## 1. Preparation

- [ ] 1.1 Ensure develop branch is clean (no uncommitted changes)
- [ ] 1.2 Fetch latest from upstream (`git fetch upstream`)
- [ ] 1.3 Review upstream changes for potential conflicts

## 2. Merge Upstream

- [ ] 2.1 Create backup branch (`git branch develop-backup`)
- [ ] 2.2 Merge upstream/master into develop
- [ ] 2.3 Resolve any merge conflicts
- [ ] 2.4 Run linting to verify merge didn't break anything

## 3. Verification

- [ ] 3.1 Run quick sanity tests
- [ ] 3.2 Verify openspec changes are still valid
- [ ] 3.3 Check that our custom code still exists

## 4. Push

- [ ] 4.1 Push updated develop to origin
- [ ] 4.2 Delete backup branch if merge successful
