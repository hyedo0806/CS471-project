
## Basic Information

This is the result of the project on cs471

* Contributer: Dohye Kim and Seonho An (alphabetical order)
* Title: GraphSAGE-based Win/Loss Prediction in League of Legends
* We use only cpu for our code

## How to setup and execute

(We do not tested yet on Windows environment)

Mac and Linux case (including Apple Silicon)

```
git clone https://github.com/hyedo0806/CS471-project

conda create -n cs471 python=3.7
conda activate cs471

pip install -r requirements.txt

cd ./CS471-project

# Our preprocessed data
gdown https://drive.google.com/uc?id=1OTsNTJ8jJ4QZZSoncKGrapM4kl-J6UtD

python GraphLOL.py
# for setting cutMode, you should insert one of the N, 1 or 2 after execute.
```

If you want to use Google Colab (=using .ipynb case), just execute overall codes on GraphLOL.ipynb.

## Tensorboard (not on Google Colab)

You can check your result on tensorboard. You can get the log file on ./runs initially.
If you want to lookup it, do

```
tensorboard --logdir=./runs
```
on CS471-project directory