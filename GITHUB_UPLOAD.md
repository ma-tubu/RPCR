# GitHub Upload Notes

Target repository:

```text
https://github.com/ma-tubu/RPCR
```

## First Upload

Run inside this cleaned release folder:

```bash
cd /path/to/RPCR_GitHub
git init
git branch -M main
git remote add origin https://github.com/ma-tubu/RPCR.git
git add .
git commit -m "Initial RPCR research preview"
git push -u origin main
```

If the remote repository already has commits, pull or clone it first, then copy
this folder's contents into that repository before committing.

## Update an Existing Clone

```bash
cd /path/to/RPCR
git status
git add .
git commit -m "Update RPCR research preview"
git push
```

## Before Pushing

Check that no data or checkpoints are staged:

```bash
git status --short
git ls-files | grep -E '(cache|datasets|train_log|\\.pt$|\\.npy$|\\.pkl$)'
```

The second command should print nothing for a clean public release.

## Notes

- `README.md` uses figures in `assets/figures/`.
- `DATASETS.md` documents the expected feature layout.
- `.gitignore` excludes raw datasets, caches, checkpoints, and logs.
