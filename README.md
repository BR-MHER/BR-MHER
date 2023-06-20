# Bias Reduction in Multi-Step Goal-Conditioned Reinforcement Learning

This repository is the official implementation of BR-MHER. 

## Requirements

To install requirements:
- Install the requirements such as *tensorflow*, *mpi4py* and *mujoco_py* using pip, besides *multi-world* should be installed from this open-source multi-task benchmark environment repo https://github.com/vitchyr/multiworld;

- Clone the repo and cd into it;

- Install baselines package
    ```bash
    pip install -e .
    ```
- Install the rest dependencies.
    ```setup
    pip install -r requirements.txt
    ```

## Training

To train the model(s) in the paper, run this command:

- Train MHER
```train
python -m  baselines.run  --env=${TASK} --num_epoch 50 --num_env 6  --n_step ${NSTEP} --mode nstep --log_path=logs/${TASK}/mher_n${NSTEP}/${SEED} --save_path=models/${TASK}/mher_n${NSTEP}/${SEED} --seed ${SEED} --n_test_rollouts 20 --scale_degree 0 --Q_lr 0.001 --hidden 256
```

- Train MHER($\lambda$)
```train
python -m  baselines.run  --env=${TASK} --num_epoch 50 --num_env 6  --n_step ${NSTEP} --mode lambda --lamb 0.7 --log_path=logs/${TASK}/mher_lambda_n${NSTEP}/${SEED} --save_path=models/${TASK}/mher_lambda_n${NSTEP}/${SEED} --seed ${SEED} --n_test_rollouts 20  --scale_degree 0 --Q_lr 0.001 --hidden 256
```

- Train MMHER
```train
python -m  baselines.run  --env=${TASK} --num_epoch 50 --num_env 6  --n_step ${NSTEP} --mode dynamic --alpha 0.5 --log_path=logs/${TASK}/br_mmher_${NSTEP}/${SEED} --save_path=models/${TASK}/br_mmher_${NSTEP}/${SEED} --seed ${SEED} --n_test_rollouts 20 --scale_degree 1 --Q_lr 0.001 --hidden 256
```

- Train TMHER
```train
python -m  baselines.run  --env=${TASK} --num_epoch 50 --num_env 6  --n_step ${NSTEP} --mode nstep --log_path=logs/${TASK}/tmher_n${NSTEP}/${SEED} --save_path=models/${TASK}/tmher_n${NSTEP}/${SEED} --seed ${SEED} --n_test_rollouts 20 --scale_degree 0 --Q_lr 0.001 --hidden 256 --truncate True
```

- Train BR-MHER
```train
python -m  baselines.run  --env=${TASK} --num_epoch 50 --num_env 6  --n_step ${NSTEP} --mode nstep --log_path=logs/${TASK}/br_mher_n${NSTEP}/${SEED} --save_path=models/${TASK}/br_mher_n${NSTEP}/${SEED} --seed ${SEED} --n_test_rollouts 20 --scale_degree 1 --Q_lr 0.001 --hidden 256 --truncate True
```

- Train BR-MHER($\lambda$)
```train
python -m  baselines.run  --env=${TASK} --num_epoch 50 --num_env 6  --n_step ${NSTEP} --mode lambda --lamb 0.7 --log_path=logs/${TASK}/br_mher_lambda_n${NSTEP}/${SEED} --save_path=models/${TASK}/br_mher_lambda_n${NSTEP}/${SEED} --seed ${SEED} --n_test_rollouts 20  --scale_degree 1 --Q_lr 0.001 --hidden 256 --truncate True
```

- Train BR-MMHER
```train
python -m  baselines.run  --env=${TASK} --num_epoch 50 --num_env 6  --n_step ${NSTEP} --mode dynamic --alpha 0.5 --log_path=logs/${TASK}/br_mmher_${NSTEP}/${SEED} --save_path=models/${TASK}/br_mmher_${NSTEP}/${SEED} --seed ${SEED} --n_test_rollouts 20 --scale_degree 1 --Q_lr 0.001 --hidden 256 --truncate True
```

## Tasks
All the used tasks can be found in the file `tasks.txt`.

<!-- ## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below). -->

<!-- ## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models. -->
<!-- 
## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it.  -->

<!-- 
## Contributing

>📋  Pick a licence and describe how to contribute to your code repository.  -->