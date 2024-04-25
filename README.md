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

Then, you need to pull the hashes from dvc:

```bash
dvc pull -R data/hashes
```

Next, generate teeth crops inside pipelines container:

```bash
python -m bin.cropsgen
```

Finally, generate COCO dataset:

```bash
python -m bin.cocogen
```

## Train

```bash
python -m bin.train --config-file configs/SwinL.yaml --num-gpus 2 OUTPUT_DIR /path/to/repo/outputs/my-sota-run
```

## Eval

First, you need to optimize the thresholds for conditions and tags:

```bash
python -m bin.optimize_thresholds --exp-name my-sota-run
```

Then, you can evaluate the model:

```bash
python -m bin.eval --exp-name my-sota-run
```

## Deployment

To deploy the model to pipelines we need to convert it to TorchScript format:

```bash
python -m bin.export --exp-name my-sota-run
```

You'll also need to copy the source code. You can do it by running:
```bash
python -m bin.pipesync -o /path/to/pipelines/src
```
