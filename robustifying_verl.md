This is my folder containing verl, the RL framework. It works, and I have learned how to use it and have built many features into it, but I still find that its not so usable. 

Here are some pain points I found and fixes I want. 

## Two seperate issues to get started on 
* sparsifying test cases in code training. 
* add config for rewards that gets threaded in and validated. 


## Some pain points I have identified

### Migrating from many `.sh` files to `yaml` for configs 


### Keeping and slightly improving validation script 
* raise error if Mmeory in GPUs is not 0
* `-y` currently not working  

### Improved run logging 
* config should be associated with checkpoint

### Continuing runs robustified 
* Improve the validation script to use check that actor/ is in resume directory if running on intended resume

### Auto backup & sparsify of checkpoints


### Support for forking runs 


### Config system for rewards


### Ambitious: watcher to determine whether run is not failing but obvious failure mode
* exit if none of the reward succeeded 


### Going back & forth to wandb 


### Building this in a test-driven way becasue crashes are expensive 
* maybe requires a dummy_verl, which has various crash / data write behaviors that we can call to simulate program behaviors which are non-trivial

### Notify me using `ntfy.sh`when something went wrong 
[DEPENDENCY] * we should abstract this into a multi-use object which lives beyond this machine  

## An outline of the solution I propose (incomplete, open to suggestions!)

### How to think about checkpoints
* I think the right way to think about checkpoints is like a DAG forest, where every directory is part of this DAG
* so there is root where we start a run
* a run can fork because (a) I have intentionally forked it, (b) it ran, failed, and I want to restart from the previous checkpoint

To do this, I think every run dir should be write-once (only a single run command can write to it, it may have mulltiple checkpoints). It should have pointers to the previous and next runs in the DAG. Is the more robust way to do this symlinks or files with txt addresses? 
[MAYBE] It would be really nice to have a way to traverse this DAG. For now it can be manualy done when needed.  

### `orchastrator.py`: main component 
I propose a main `orchastrator.py` class which takes in some orchastration config and the config for the run and starts a few processes which should be watching out for the run. 
In terms of run config, it should take 
* base config -- this is a base yaml config for the run (e.g. Coding run)
* specialized config -- this is a config including changes (e.g. )
    * u can see a base config here (/shared/matan/code/pass_at_k/verl/trainer/config/ppo_trainer.yaml) and using the config (examples/grpo_trainer/run_deepseek7b_llm_math_megatron.sh). probably prefer json5 format instead of yaml but don't know if its supported by Hydra the config parser? 
* some more settings which will be mentioned throughout -- prefer giving path to json file rather then cmdline args

It should run 
* derive a path for the log dir and fill it with the necessery metedata
* run `validate_env.py`
* start a process which will sparsify checkpoint files in that dir  



In addition, I think we should maintain some database of notable runs which made it. This will be just a list of pointers to log dirs, just to organize results. Here is kinda maybe what I am thinking. 
[Human dependency -- design before running this to make sure it would work with this]


(also very much okay to give up on the more ambitious of these features and start with MVP)
