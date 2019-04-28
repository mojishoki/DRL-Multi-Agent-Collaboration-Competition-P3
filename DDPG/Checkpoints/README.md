Here, you will find the weights for a two brain DDPG which uses the model with 512-256 hidden layer size and dropout, plus the weights (solved in 482 episodes) of the single brain DDPG with the same configurations. 

The two other checkpoints are for a single brain DDPG model with hidden layer size 400-300 and _without_ dropout. Therefore, to load the weights, you will need to manually change the hidden layer sizes and comment the `self.dropout()` part of the `forward` passes of actor and critic in `model.py`.

The name of each file describes their hyperparameters. We refer to `DDPG_train.py` as to which order the hyperparameters are listed:

```python
{agents[0].batch_size}{agents[0].buffer_size}{agents[0].gamma}{agents[0].tau}{agents[0].lr_actor}{agents[0].lr_critic}{agents[0].weight_decay}{agents[0].random_seed}{agents[0].update_every}{agents[0].learn_num}{agents[0].noise.sigma}{agents[0].noise.theta}'
```

You can plot the results of any of the above runs using the `plot_results()` function in the notebook `DDPG_main.ipynb`. The first argument can be any of the `...max` score history above and the second should be `DDPG`.