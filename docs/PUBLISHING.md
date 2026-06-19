# Publishing To GitHub

This repository is set up to publish source code, configuration, and recipes without committing local datasets, generated outputs, model checkpoints, logs, or archives.

## Before Publishing

1. Keep model files such as `*.pt`, `*.pyth`, `*.pth`, `*.ckpt`, and `*.ts` out of Git.
2. Keep EPIC-KITCHENS videos, extracted audio, HDF5 files, and generated CSV/JSON outputs out of Git.
3. Put large required assets in GitHub Releases, cloud storage, or Git LFS, then link to them from the README.
4. Add a license file before making the project public if you want other people to reuse the code.

## Check What Will Be Published

```powershell
git status --short
git ls-files
```

If generated files are already tracked, remove them from Git without deleting your local copies:

```powershell
git rm --cached P01_01_active_regions.csv P01_01_audioinceptionnext_ann.csv P01_01_ssast_active.csv P01_01_ssast_ann.csv P01_01_ssast_ann_autogen.csv P01_01_train_regions_pred.csv P1_01_pann.csv Visualize_Dataset/P01_01_vs_train.csv
git rm --cached -r AV_EventRecognition_export_bundle_20260327 AV_EventRecognition_runtime_assets_cucumber_20260327
```

## Publish

Create an empty GitHub repository first, then run:

```powershell
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git add .gitattributes .gitignore .gitmodules .github/workflows/smoke.yml docs/PUBLISHING.md README.md AV_EventRecognition/recipes
git commit -m "chore: prepare repository for GitHub publishing"
git push -u origin main
```

For a fresh clone with the `AudioInceptionNeXt` submodule:

```powershell
git clone --recurse-submodules https://github.com/<your-user>/<your-repo>.git
```
