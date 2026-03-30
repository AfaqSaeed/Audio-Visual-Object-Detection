param(
    [string]$BundleRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) "AV_EventRecognition_export_bundle_20260327"),
    [string]$Video = "D:\datasets\EPIC-KITCHENS\P01\videos\P01_01.MP4",
    [string]$AudioHdf5 = "D:\datasets\hdf5_from_wav\wav_to_pickled.hdf5",
    [string]$CondaEnv = "avod_py171"
)

$assetRoot = (Resolve-Path $PSScriptRoot).Path
$bundleRoot = (Resolve-Path $BundleRoot).Path

conda run -n $CondaEnv python (Join-Path $bundleRoot "AV_EventRecognition\run_av_event_pipeline.py") `
  --video $Video `
  --audio-csv (Join-Path $bundleRoot "AV_EventRecognition\outputs\bundle_independence_cutting_cucumber\P01_01_ssast_ann_autogen.csv") `
  --audio-output-summary-csv (Join-Path $bundleRoot "AV_EventRecognition\outputs\bundle_independence_cutting_cucumber\P01_01_ssast_ann_autogen_summary.csv") `
  --auto-generate-audio-csv `
  --audio-checkpoint (Join-Path $assetRoot "SSAST_EPIC_SOUNDS.pyth") `
  --audio-hdf5 $AudioHdf5 `
  --audio-video-id "P01_01" `
  --audio-window-mode annotation_regions `
  --audio-annotations-file (Join-Path $bundleRoot "epic-sounds-annotations\EPIC_Sounds_train.csv") `
  --recipe-json (Join-Path $bundleRoot "AV_EventRecognition\recipes\cutting_cucumber.json") `
  --output-dir (Join-Path $bundleRoot "AV_EventRecognition\outputs\bundle_independence_cutting_cucumber") `
  --model (Join-Path $assetRoot "yoloe-11s-seg.pt") `
  --world-model (Join-Path $assetRoot "yolov8m-world.pt") `
  --fastsam-model (Join-Path $assetRoot "FastSAM-s.pt") `
  --max-clips 1
