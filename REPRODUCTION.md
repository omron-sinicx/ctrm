# Reproduction of Experiments

You can download MAPP demonstrations, benchmarks, and learned models from [ctrm_data.zip]().

## Step 1. Generate MAPP Demonstrations

```sh
bash scripts/exp_scripts/data_gen_hetero.sh    # required time: half day by x40 multiprocessing, seed=100000
```

The data will be stored in `/data/demonstrations/learn_hetero/21-30`.

## Step 2. Model Training

```sh
bash scripts/exp_scripts/learn_hetero.sh cuda:0      # required time: 2 hours, you can use cpu instead of cuda
```

The repo includes the trained model in `/workspace/trained_model/aamas22-main`.

## Step 3. Generate Benchmarks

```sh
bash scripts/exp_scripts/benchmark_gen_hetero.sh   # seed=46
```

The data will be stored in `/data/benchmark`.

## Step 4. Evaluation

All the results will be saved in `/data/exp`.

```sh
bash scripts/exp_scripts/eval_ctrm_large_learned_ind.sh    # required time: 1 day
```

The used trained model is `ctrm_data/models/with_ind_k15`.

### Baselines

#### random (equivalent to a simplified version of PRM [1])

```sh
bash scripts/exp_scripts/eval_random_large.sh
```

#### grid

```sh
bash scripts/exp_scripts/eval_grid_large.sh
```

#### SPARS [2]

```sh
bash scripts/exp_scripts/eval_spars_large.sh
```

In the heterogeneous scenario, the method uses multiprocessing. Although this affects runtime, the method anyway results in a low success rate (hence excluded in the figures with quality metrics).

#### square (rect)

```sh
bash scripts/exp_scripts/eval_square_large.sh
```

</details>

### Ablation Study

#### Model Training

```sh
bash scripts/exp_scripts/learn_hetero_wo_comm.sh         # without communication
bash scripts/exp_scripts/learn_hetero_wo_indicator.sh    # without indicator
```

The trained models are already included in `ctrm_data/models`.

#### Evaluation

```sh
bash scripts/exp_scripts/eval_ctrm_ablation_learned_ind.sh
```

## Reference

1. Karaman, S., & Frazzoli, E. (2011). Sampling-based algorithms for optimal motion planning. The international journal of robotics research (IJRR)
2. Dobson, A., Krontiris, A., & Bekris, K. E. (2013). Sparse roadmap spanners. In Algorithmic Foundations of Robotics X.
