# Bias Resilient Multi-Step Off-Policy Goal-Conditioned Reinforcement Learning

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
python -m  baselines.run  --env=${TASK} --num_epoch 50 --num_env 6  --n_step ${NSTEP} --mode nstep  --log_path=logs/${TASK}/mher_n${NSTEP}/${SEED} --save_path=models/${TASK}/mher_n${NSTEP}/${SEED} --seed ${SEED} --n_test_rollouts 20 --Q_lr 0.001 --pi_lr 0.001 --tau 0.5 --delta 10 --use_huber True --truncate False --policy_delay 2 --noise_std 0
```

- Train MHER($\lambda$)
```train
python -m  baselines.run  --env=${TASK} --num_epoch 50 --num_env 6  --n_step ${NSTEP} --mode lambda --lamb 0.7  --log_path=logs/${TASK}/mher_n${NSTEP}/${SEED} --save_path=models/${TASK}/mher_n${NSTEP}/${SEED} --seed ${SEED} --n_test_rollouts 20 --Q_lr 0.001 --pi_lr 0.001 --tau 0.5 --delta 10 --use_huber True --truncate False --policy_delay 2 --noise_std 0
```

- Train MMHER
```train
python -m  baselines.run  --env=${TASK} --num_epoch 50 --num_env 6  --n_step ${NSTEP} --mode dynamic --alpha 0.5 --log_path=logs/${TASK}/mher_n${NSTEP}/${SEED} --save_path=models/${TASK}/mher_n${NSTEP}/${SEED} --seed ${SEED} --n_test_rollouts 20 --Q_lr 0.001 --pi_lr 0.001 --tau 0.5 --delta 10 --use_huber True --truncate False --policy_delay 2 --noise_std 0
```

- Train TMHER
```train
python -m  baselines.run  --env=${TASK} --num_epoch 50 --num_env 6  --n_step ${NSTEP} --mode nstep --log_path=logs/${TASK}/mher_n${NSTEP}/${SEED} --save_path=models/${TASK}/mher_n${NSTEP}/${SEED} --seed ${SEED} --n_test_rollouts 20 --Q_lr 0.001 --pi_lr 0.001 --tau 0.5 --delta 10 --use_huber True --truncate True --policy_delay 2 --noise_std 0
```

- Train TMHER($\lambda$)
```
python -m  baselines.run  --env=${TASK} --num_epoch 50 --num_env 6  --n_step ${NSTEP} --mode lambda --lamb 0.7  --log_path=logs/${TASK}/mher_n${NSTEP}/${SEED} --save_path=models/${TASK}/mher_n${NSTEP}/${SEED} --seed ${SEED} --n_test_rollouts 20 --Q_lr 0.001 --pi_lr 0.001 --tau 0.5 --delta 10 --use_huber True --truncate True --policy_delay 2 --noise_std 0
```

- Train QR-MHER

```
python -m  baselines.run  --env=${TASK} --num_epoch 50 --num_env 6  --n_step ${NSTEP} --mode lambda --lamb 0.7  --log_path=logs/${TASK}/mher_n${NSTEP}/${SEED} --save_path=models/${TASK}/mher_n${NSTEP}/${SEED} --seed ${SEED} --n_test_rollouts 20 --Q_lr 0.001 --pi_lr 0.001 --tau 0.75 --delta 10 --use_huber True --truncate False --policy_delay 2 --noise_std 0
```

- Train BR-MHER
```train
python -m  baselines.run  --env=${TASK} --num_epoch 50 --num_env 6  --n_step ${NSTEP} --mode lambda --lamb 0.7  --log_path=logs/${TASK}/mher_n${NSTEP}/${SEED} --save_path=models/${TASK}/mher_n${NSTEP}/${SEED} --seed ${SEED} --n_test_rollouts 20 --Q_lr 0.001 --pi_lr 0.001 --tau 0.75 --delta 10 --use_huber True --truncate True --policy_delay 2 --noise_std 0
```

NOTE `tau` controls the quantile level in the quantile regression.

## Tasks
All the used tasks can be found in the file `tasks.txt`.
