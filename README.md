*Pytorch Implementation of <[Energy-Based Imitation Learning](https://arxiv.org/abs/2004.09395)>*

**Important Notes**

This repository is based on [rlswiss](https://github.com/KamyarGh/rl_swiss), which is an extension from the August 2018 version of [rlkit](https://github.com/vitchyr/rlkit). Since then the design approaches of rlswiss and rlkit have deviated quite a bit, and it is for this reason that we are releasing rlswiss as a separate repository. *If you find this repository useful for your research/projects, please cite this repository as well as [rlkit](https://github.com/vitchyr/rlkit).*

# Algorithms

Implemented RL algorithms:
- Soft-Actor-Critic (SAC)

Implemented LfD algorithms:
- Adversarial methods for Inverse Reinforcement Learning
    - AIRL / GAIL / FAIRL / Discriminator-Actor-Critic
- Behaviour Cloning
- DAgger
- EBIL

# How to run

Notes:
- First appropriately modify rlkit/launchers/config.py
- run_experiment.py calls srun which is a SLURM command. You can use the `--nosrun` flag to not use SLURM and use your local machine instead.
- The expert demonstrations used for imitation learning experiments can be found at [THIS LINK](https://drive.google.com/drive/folders/1jwKb5FjFtAlvBUDdHiHJN0i7PsBCthfg?usp=sharing). To use them please download them and modify the paths in expert_demos_listing.yaml.
- The yaml files describe the experiments to run and have three sections:
..* meta_data: general experiment and resource settings
..* variables: used to describe the hyperparameters to search over
..* constants: hyperparameters that will not be searched over
- The conda env specs are in rl_swiss_conda_env.yaml. You can refer to [THIS LINK](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#create-env-from-file) for notes on how to set up your conda environment using the rl_swiss_conda_env.yaml file.
- You need to have Mujoco and mujoco-py installed.
- Due to a minor dependency on rllab, you would have to also install rllab. I will try to remove this dependency in future versions. The dependency is that run_experiment.py calls build_nested_variant_generator which uses something from rllab.

## Reproducing Imitation Learning Results
### Training Expert Policies
If you would like train your own expert policies with Soft-Actor-Critic you can for example run:
```bash
python run_experiment.py --nosrun -e exp_specs/sac.yaml
```
To train a policy for a different environment, add your environment to the file in rlkit/envs/envs_dict and replace the name of your environment in the env_specs->env_name field in sac.yaml.

### Generating Demonstrations Using an Expert Policy
Modify exp_specs/gen_exp_demos.yaml appropriately and run:
```bash
python run_experiment.py --nosrun -e exp_specs/gen_exp_demos.yaml
```
Put the path to your expert demos in the expert_demos_listing.yaml file.

### Normalizing the Demonstrations
Use scripts/normalize_exp_demos.py to normalize the generated demonstrations. As of the time of this writing this script does not take in command-line arguments. In line 108, put the name yo uused for your new entry in the expert_demos_listing.yaml file. Put the path to the normalized expert demos in the expert_demos_listing.yaml file.

### Training F/AIRL
Modify exp_specs/adv_irl.yaml appropriately (specifically, do not forget to replace the name of the normalized expert demos with the name of your new entry in expert_demos_listing.yaml) and run:
```bash
python run_experiment.py --nosrun -e exp_specs/adv_irl.yaml
```
For all four imitation learning domains we used the same hyperparameters except grad_pen_weight and reward_scale which were chosen with a small hyperparameter search.

### Training BC
```bash
python run_experiment.py --nosrun -e exp_specs/bc.yaml
```

### Training DAgger
```bash
python run_experiment.py --nosrun -e exp_specs/dagger.yaml
```

### Training EBIL
```bash
python run_experiment.py --nosrun -e exp_specs/deen/deen_hopper_4.yaml
```
Note: sigma is the only hyperparameter to tune, which is a rather important parameter to train a good energy model. DEEN can be replaced by any powerful algorithm in any deep learning framework to train an energy model, another choice can be ncsn[https://github.com/ermongroup/ncsn]

```bash
python run_experiment.py --nosrun -e exp_specs/ebil/ebil_hopper_4.yaml
```
Note: set sigma as the trained energy model. Remember to keep the expert demos the same as the one in DEEM trainining.

## Reproducing State-Marginal-Matching Results
### Generating the target state marginal distributions
This is a little messy and I haven't gotten to cleaning it up yet. All the scripts that you see in the appendix of the paper can be found in these three files: data_gen/point_mass_data_gen.py, data_gen/fetch_state_marginal_matching_data_gen.py, data_gen/pusher_data_gen.py.

### Training SMM Models
When you run any of these scripts, at every epoch in the log directory of the particular experiment images will be save to demonstrate the state distribution of the policy obtained at that point. For more information about the visualizations you can have a look at the log_visuals function implemented for each of the environments used in the state-marginal-matching experiments. 

The SMM data is generated by hand designed points.

```bash
python run_experiment.py --nosrun -e exp_specs/ebil/ebil_point_mass_square.yaml
```
```bash
python run_experiment.py --nosrun -e exp_specs/ebil/ebil_point_mass_triangle.yaml
```

Plot SMM energy.

```bash
python run_experiment.py --nosrun -e exp_specs/ebil/plot_point_mass_square.yaml
```
```bash
python run_experiment.py --nosrun -e exp_specs/ebil/plot_point_mass_triangle.yaml
```
