# Actor-Critic model for gym's BipedalWalker environment.

The code is based on a post of Yash Patel.

URL: https://towardsdatascience.com/reinforcement-learning-w-keras-openai-actor-critic-models-f084612cfd69
I changed some logical details and chose pytorch instead of tensorflow, also environment differs, that required changes as well.


# Usage

To start training the agent (its better to increase number of trials, at the moment weights are not suitable):

```
from AC import main
main(num_trials=1000, trial_len=500)
```

To see the agent walking:

```
from AC import walk
walk(num_trials=10, trial_len=500)/
```

#WARNING 

Weights are not perfect now. There are two possible reasons:
1) not enough learning iterations (1000 trials)
2) there som mistakes in model

