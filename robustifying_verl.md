This is my folder containing verl, the RL framework. It works, and I have learned how to use it and have built many features into it, but I still find that its not so usable. 

Here are some pain points I found and fixes I want. 

## Two seperate issues to get started on 
* sparsifying test cases in code training. (want to add config first)
* add config for rewards that gets threaded in and validated. 

### Update to my verl: threading of reward config
* currently, when I want to make a slightly tweaked reward of another reward, I simply make a new function and name it with the config. (see APPS_reward.py). This is a bit painful. 
* I want to modernize my setup with a reward config that is passed around.
* The path by which I call my reward (do not go crazy exploring other stuff) is e.g.  /shared/matan/code/pass_at_k/custom/run_scripts/02/17/INCREASE_PENALTY_320startindex_oss120_cont_s2to3_training_hidden_w_gstep.sh, to main_ppo, which calls trainer.py, somewhere in there instantiates `naive.py` + pickles my reward function with extra configs (?) -- i think the code for this as the top of `naive.py`. This means that the function needs to be pickle-able, so i don't think it can use global variables / be a method of a class, etc. 
* naive.py constrains that all reward functions must take `data_source: str, solution_str: str, ground_truth: Any, extra_info: dict` -- I think this `extra_info` is one time global which gets curried away so maybe its the place where we can thread in configs. 
* to understand which kind of configs I might want to pass in, see APPS_reward.py function names which essentially bake in the config.
* one way I am thinking about passing in my config is thru this extra reward options, which would effectively select which subfunction to run. Not sure whether this is the right way? 
    * I am also thinking about building an orchastrator for verl, and this orchastrator could take the reward options and then pick the correct reward name. This might be the right option if I want minimal code reuse... 
    * I am not sure what is the right balance: on one end we have mono functions with many if sttaements based on config, on the other no codereuse -- what do u think are some options to find balance, looking at APPS_reward.py as an example progression of tweaking reward functions?     

I understand exploring is useful, but this is a large monorepo and many of the paths and files here are useless to me and do not need to be explored. 

# Improving verl

## Some pain points I have identified
Here is a scatter of some pain points identified and the plan below is supposed to solve

### Migrating from many `.sh` files to `yaml` for configs 
(see example subfolder in `/shared/matan/code/pass_at_k/custom/run_scripts/02/17` for the way I am calling right now)

### Improved run logging 
* config should be associated with checkpoint and not a set of checkpoint
(solved below by single run per log directory)

### Continuing runs robustified 
* Improve the validation script to use check that actor/ is in resume directory if running on intended resume

### Auto backup & sparsify of checkpoints
* always keep the medata + global_step directories 

### Support for forking runs 
* will be in the config for 

### Ambitious: watcher to determine whether run is not failing but obvious failure mode
* exit if none of the reward succeeded 


### Going back & forth to wandb 
* this one is a bit difficult , would be nice ot have the ID in each run and the rollout directory added in wandb. I think that should work. 
* would be nice to open a view for a run given all the urls -- is there a way to do that in wandb.  

### Building this in a test-driven way becasue crashes are expensive 
* maybe requires a dummy_verl, which has various crash / data write behaviors that we can call to simulate program behaviors which are non-trivial. 

### Notify me using `ntfy.sh`when something went wrong 
[DEPENDENCY] * we should abstract this into a multi-use object which lives beyond this machine. 

# An outline of the solution I propose (incomplete, open to suggestions!)
I propose to put all this in `vfh/` (verl for humans -- and LLMs ofc 😉) folder.  

### Changing how we save data (checkpoints + rollouts + metadata)
* I think the right way to think about saving data is that each run (defined as one time we run verl, with no interaptions) has its own dir. The dir contains `checkpoints/` which has the `global_step_{N}` and `rollouts/` which has `train/` and `val` with `jsonl` for each step. 
* every directory is part of this DAG of runs. 
* a run can fork because (a) I have intentionally forked it, (b) it ran, failed, and I want to restart from the previous checkpoint. 
* each run dir links to the previous and next run (next run link is added once it starts). Maybe we symlink? 
* I think we can have `runs/{month}/{day}/{not super sure on naming convention? Maybe nice to include ROOT if its a root run}_{hr}_{min}/`
* runs should contain all the metadata which was passed in to the orchastrator. 
* runs should have an id and link to wandb which is `https://wandb.ai/matan-shtepel-carnegie-mellon-university/subtle_reasoning_repro/runs/ocybtbz9` (where we have organization/project/id) -- we should generate the id and pass it in `verl/utils/tracking.py` (instead of letting it get autodetermined). 


## `orchastrator.py`: main component 
I propose a main `orchastrator.py` class which takes in some orchastration config and the config for the run and starts a few processes which should be watching out for the run. 
In terms of run config, it should take 

Orchastrator should have its own config. Specified by a dataclass. Probably the best way to go about this is u ask me what I think should be in the config and I lyk. Some things can just be hardcoded for now. 

### Starting new runs 

#### For starting a new run
* base config -- this is a base yaml config for the run (e.g. Coding run)
* specialized config -- this is a config including changes applied atop the base config. 
    * u can see a base config here (/shared/matan/code/pass_at_k/verl/trainer/config/ppo_trainer.yaml) and using the config (examples/grpo_trainer/run_deepseek7b_llm_math_megatron.sh). probably prefer json5 format instead of yaml but don't know if its supported by Hydra the config parser? 
* some more settings which will be mentioned throughout -- prefer giving path to json file rather then cmdline args
* `-y` for allowing validation to happen without quitting and `-yy` for skipping validation altogether. 

#### For continuing a run  
should have some simple options like 
* specifying the dir
* specifying new configs (optional) -- some sanity checks. 


#### Should always do 
* run `validate_env.py`
- check parsing of the config files (can we use json instead of yaml?)


### Spawn deamons 
May add more deamons later which checkup on run status and e.g. ping me on ntfy, but I think this is good enough. Which deamons (is this the right term btw?) are running should be configurable. 

#### Checkpoint save and delete
* i save checkpoints every 40 steps in case of crashes. 
* i actually care to backup every 80 steps. 
* i want a process that spawns with the run and will wake up every so often to sparsify and backup checkpoints. 
* my backup system is via `dvc add` then `dvc push`
* Once things are backed up (would be nice to do some confirmation -- I think I have some logic for that in `$SHARED_HOME/bin/s3-dvc-tools/`) can remove the contents of dir, keep the empty dir to lmk the checkpoint existed. 

(for now let's keep all the rollouts but later might want to remove rollouts)

## Helper functions (not part of orchastrator to seperate concerns)

### Tree traverser 
not super sure what I want this feature to look like, but here are some things that seem useful. 
* Allows u to traverse the DAG starting from a certain root node interactively (?) and then 
    - add the DAG to some list of named, notable runs. 
    - view the DAG in wandb 
        - claude how to desc: Write a Python function `make_wandb_url(entity: str, project: str, run_ids: list[str]) -> str` that returns a wandb workspace URL with a JSON filter in the query params so only the specified runs are visible. The filter format is: {"$and": [{"name": {"$in": run_ids}}]} — URL-encode it and append as ?filters=<encoded>.
- filter runs by config values and number of saved checkpoints (by default requires at least one `checkpoints/` subdir to indicate that the run did not immidately crash)
- allow filtering down to the "notable" runs. 

### Dummy verl 
* a dummy verl script that can write fake checkpoints, crash, wait, etc, so we can test feautures. 


-------------
Anything I am missing, which seems confusing? 

I think the right thing to do is to incrmentally build this feature, since its quite a large feature, and incrementally test between starting new features. I will test some of the more interactive features since I think ur still a bit clumsy / token wasteful at that. 

Any pain points I am not addressing? Suggestions you'd make? 