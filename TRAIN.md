## Demo
For hard-coded policy demo, change `all_args` at the bottom of `MAZFish.py` as needed and run `python MAZFish.py`

## Training and evaluation
For a single training/eval run (not on cluster), modify the parameters and flags in `run_lambda_single.sh` and run it. This runs training, evaluation, postprocessing, and generates figures in a subfolder under `onpolicy/custom/forage/results/`

For running multiple training/evals in parallel:
1. Modify the `SWEEP_CONFIGS` array in `sweep_configs.py` to contain all the flags/parameters to sweep over. For boolean flags, only the key matters, value can be arbitary. For numerical flags, the key should be the flag name and the value should be the number.
2. Run `python sweep_configs.py` to print the number of runs that should be performed to sweep over the chosen values. Change the `#SBATCH --array=0-x` line in `batch_train_sweep.sh` depending on this number (0-indexed so subtract one).
3. Modify `run_single_train.sh` to change the number of training steps, number of agents, perception type (note that currently only `"projected"` is supported -- `"rays"` will lead to some `NotImplemented` errors), number of threads, episode length, number of eval episodes per thread, etc. and if some of the default training features should be changed (binocular_depth_only, use_1dof_eyes, binocular_angle_only) from the `python train_ZFish.py` line. Also set the evaluation food density/eating distribution decay constant in the `python eval_ZFish.py` line as needed.
4. Run `sbatch batch_train_sweep.sh`. Check `logs/MAZFish_cluster_test_0` for all log/err files. Each run folder will be added to `results/rmappo-MultiAgentForagingEnv-check` named with the `run_name` passed in `sweep_configs.py`, with all evaluation episodes and figures.
5. To check for training degeneracy across multiple seeds, in `analyze_training_degen.py` change the `folder_contains` list at the top to contain a list of substrings in each run that you want to compare. For each list in `folder_contains`, the script will check for a run folder that has all substrings in that list, and use this to find a collection of run folders. Running `analyze_training_degen.py` will generate some plots comparing the AUC/average angle metrics to the average number of eating events and average reward across all eval episodes, allowing to check for training degeneracy and selecting the best performing training runs.

IMPORTANT NOTE: currently there is a mismatch between `cfg.py` and the args that are saved in the results' log folder's `env_args.json` -- this causes the function `get_old_cfg_args` to not update all `cfg` args during evaluation, so the current parameters in `cfg.py` do play a role in evaluation. This is a bug that needs to be fixed, but as long as `cfg.py` has the correct `AGENT_PARAMS["food_detection_range"]` (and some other arguments held constant) at all times, this should not be a problem; however, keep this in mind when adding more `cfg`/`all_args` features in the future.

Also note that all files not mentioned in this README and not used in any of the bash scripts mentioned in this README are old and should not be used.

## Generating report-style figures

The figures generated via the above training/eval are for debugging/intuitions only; to generate the final report-style figures, change the `outputs_folder` at the top of `report_movement.ipynb`, `report_vergence.ipynb`, and `report_rnn.ipynb` to the correct run's outputs folder, and run all notebooks in interactive mode to generate the figures.

Also, `agent_setup_figures.py` is used to generate parts of Figure 1 in the manuscript (the distribution of agent locations after 1 bout, the action space triangle, and the eating probability distribution).
