# Pano Patho MaskDINO

## Docker

```bash
make sync-submodules
JUPYTER_PORT=<your choice> make main-jupyter
```

## Dataset generation
MaskDINO implementation requires a dataset in COCO format.
First, you need to download the data from LakeFS ioxray repository to `data/raw` directory.
Here is the detailed [instruction](https://diagnocat.atlassian.net/wiki/spaces/RES/pages/310837418/New+DataOps+Practices+2D+Data+Handling+Description) of how to do it:

Next, generate teeth crops inside pipelines container:

```bash
python -m bin.cropsgen
```

Then, generate COCO dataset:

```bash
python -m bin.cocogen
```

## Train

```bash
python -m bin.train --config-file configs/SwinL.yaml --num-gpus 1 OUTPUT_DIR /path/to/repo/outputs/my-sota-run
```
