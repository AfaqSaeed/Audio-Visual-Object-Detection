# Audio-Visual Event Recognition

Disclaimer: Parts of this repository, including code, documentation, and workflow setup, were developed with AI assistance using Codex. Please review and validate the implementation before using it in research or production settings.

This repository contains an audio-visual event recognition pipeline built on:

- EPIC-SOUNDS audio event detection
- prompted visual detection with YOLO-World + FastSAM
- simple event rules defined in JSON recipe files

The main pipeline lives in `AV_EventRecognition/run_av_event_pipeline.py`.

## Minimum Required Files

You need these inputs before running the pipeline:

- A video file
  - Example: `path\to\videos\P01_01.MP4`
- An HDF5 file created from extracted audio
  - Example: `path\to\audio\wav_to_pickled.hdf5`
- An event recipe JSON file
  - Example: `AV_EventRecognition/recipes/cutting_cucumber.json`

You also need these model/config files:

- Audio inference script
  - `epic-sounds-annotations/src/tools/infer_hdf5_video.py`
- Audio config
  - `epic-sounds-annotations/src/configs/EPIC-Sounds/ssast/SSAST_b_vit_p16.yaml`
- Audio checkpoint
  - `SSAST_EPIC_SOUNDS.pyth download from here` [SSAST_EPIC_SOUNDS.pyth](https://www.dropbox.com/scl/fi/3luxvggu30smj9xxgsrjg/SSAST_EPIC_SOUNDS.pyth?rlkey=qzygp59f4joq8q4r586r33t6t&e=2&dl=0)
- Visual inference script
  - `ApplyYOLOE/predict_yoloe_seg_video.py`
- Visual model file names
  - `yolov8l-world.pt - this the ultralytics module will download itself - you also have other options depending on processing power yolov8s-world.pt << yolov8m-world.pt << yolov8x-world.pt`  
  - `FastSAM-x.pt - this will also be downlaoded by ultralytics you can also specify FastSAM-s.pt` 

The current AV pipeline defaults use the `fastsam-world` backend, so the main visual weights you need are:

- a YOLO-World checkpoint name
- a FastSAM checkpoint name

## Conda Environment

Create conda environment using the following commands

```powershell

conda env create -f environment_avod_py171.yml
conda activate avod_py171

```

Run everything from the `avod_py171` environment.

Example:

```powershell
conda run -n avod_py171 python ...
```

## Audio Extraction Placeholder

Audio extraction and HDF5 creation should be done before this pipeline.

1. Extract audio from the videos by running:
  ```
  python audio_extraction/extract_audio.py /path/to/video /output/path 
  ```
2. Save audio in HDF5 format by running:
  ```
  python audio_extraction/wav_to_hdf5.py /path/to/extracted/audio /output/hdf5/wav_to_hdf5_audio.hdf5
  ```

## Basic Run

This command will:

1. generate the audio CSV from the HDF5 file
2. find the audio event from the recipe
3. extract the matching video clip
4. run visual detection
5. apply the AV event rule

Example:

```powershell
conda run -n avod_py171 python AV_EventRecognition/run_av_event_pipeline.py `
  --video "path\to\videos\P01_01.MP4" `
  --audio-csv "P01_01_ssast_ann_autogen.csv" `
  --audio-output-summary-csv "P01_01_ssast_ann_autogen_summary.csv" `
  --auto-generate-audio-csv `
  --audio-cfg "epic-sounds-annotations\src\configs\EPIC-Sounds\ssast\SSAST_b_vit_p16.yaml" `
  --audio-checkpoint "SSAST_EPIC_SOUNDS.pyth" `
  --audio-hdf5 "path\to\audio\wav_to_pickled.hdf5" `
  --audio-video-id "P01_01" `
  --audio-window-mode annotation_regions `
  --audio-annotations-file "epic-sounds-annotations\EPIC_Sounds_train.csv" `
  --recipe-json "AV_EventRecognition\recipes\cutting_cucumber.json" `
  --output-dir "AV_EventRecognition\outputs\cutting_cucumber" `
  --max-clips 1
```

## Sample: Chopping Cucumber On P01_01

This is the exact command used to verify the exported bundle on `P01_01`. Run it from the bundle root after unzipping:

```powershell
conda run -n avod_py171 python AV_EventRecognition\run_av_event_pipeline.py `
  --video "D:\datasets\EPIC-KITCHENS\P01\videos\P01_01.MP4" `
  --audio-csv "AV_EventRecognition\outputs\bundle_independence_cutting_cucumber\P01_01_ssast_ann_autogen.csv" `
  --audio-output-summary-csv "AV_EventRecognition\outputs\bundle_independence_cutting_cucumber\P01_01_ssast_ann_autogen_summary.csv" `
  --auto-generate-audio-csv `
  --audio-checkpoint "D:\Audio-Visual Object Detection\SSAST_EPIC_SOUNDS.pyth" `
  --audio-hdf5 "D:\datasets\hdf5_from_wav\wav_to_pickled.hdf5" `
  --audio-video-id "P01_01" `
  --audio-window-mode annotation_regions `
  --audio-annotations-file "epic-sounds-annotations\EPIC_Sounds_train.csv" `
  --recipe-json "AV_EventRecognition\recipes\cutting_cucumber.json" `
  --output-dir "AV_EventRecognition\outputs\bundle_independence_cutting_cucumber" `
  --model "D:\Audio-Visual Object Detection\yoloe-11s-seg.pt" `
  --world-model "D:\Audio-Visual Object Detection\yolov8m-world.pt" `
  --fastsam-model "D:\Audio-Visual Object Detection\FastSAM-s.pt" `
  --max-clips 1
```

## Output Files

The pipeline writes results into the output directory you provide.

Typical output structure:

- `clips/`
- `rendered/`
- `predictions/`
- `reports/`
- `summary.json`

## Event Recipe JSON

Each event is defined by a JSON recipe file in `AV_EventRecognition/recipes`.

A recipe tells the pipeline:

- which audio label to look for
- what final event name to assign
- which visual prompts to run
- which visual classes must be present

The visual prompts in the recipe are used with the YOLO-World + FastSAM detection stage.

Example:

```json
{
  "audio_label": "cut / chop",
  "positive_event_name": "chopping cucumber",
  "fallback_event_name": "other chopping event",
  "prompts": ["cutting board", "knife", "cucumber", "food other than cucumber"],
  "must_have": [
    ["cutting board", "chopping board"],
    ["knife"],
    ["cucumber"]
  ],
  "conf": 0.01,
  "distance_threshold_ratio": 0.06,
  "min_support_frames": 3,
  "clip_pad_before": 1.0,
  "clip_pad_after": 1.0
}
```

### Required Recipe Fields

- `audio_label`
- `positive_event_name`
- `fallback_event_name`
- `prompts`
- `must_have`

### Notes On `must_have`

Each entry in `must_have` is one required class group.

Examples:

- `["knife"]`
- `["cutting board", "chopping board"]`
- `["food", "cucumber"]`

This means:

- one detection from each group must be found
- if a group has multiple names, any one of them can satisfy that group

## Existing Example Recipes

- `AV_EventRecognition/recipes/cutting_cucumber.json`
- `AV_EventRecognition/recipes/chopping_celery.json`
- `AV_EventRecognition/recipes/washing_carrot.json`
- `AV_EventRecognition/recipes/washing_cucumber.json`

## Visualization

To visualize one report as an evidence video:

```powershell
conda run -n avod_py171 python AV_EventRecognition/visualize_event_evidence.py `
  --report-json "AV_EventRecognition\outputs\cutting_cucumber\reports\hit_00031_0108_277_0109_074.json"
```

Disclaimer: Parts of this repository, including code, documentation, and workflow setup, were developed with AI assistance using Codex.
