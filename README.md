# rpsskynet

### Rock Paper Scissors Skynet - Pytorch

Welcome to Rock, Paper, Scissors Skynet for Pytorch!

To install this repo:

```
pip install git+https://github.com/CerebralSeed/rpsskynet.git#egg=rpsskynet
```

It WILL install Pytorch, if you do not have that already, which is the only dependency. 

Once installed, to run your RPS adversary, use the following Python script:

```
from rpsskynet.rpsskynet import rps_skynet

rps_skynet()
```

To save/load a previous model, just use the prompts after the above is running. Type `save` or `load` and it will ask you for the file path. If it's in the same file path as the above run script you created, just type the file name only. A slightly pretrained model the size of a tardigrade's brain is inside of the pretrained folder on the Github repo page. 

If you are starting to train a model from scratch, be sure to play at least 200 games to initialize it's parameters. Then it should start performing better. 

This was developed for fun and can be used as a training tool for new ML learners. The only file in the repo is under 130 lines of code.

Here is a short example run:

```
Welcome to Rock, Paper, Scissors Skynet! Please enter my hidden_size(i.e. 32):32
Please enter the number of mid_layers(Hint: 2; do not exceed 8 layers to avoid the vanishing gradients problem):2
Model size is 2723 parameters spread out over 67 neurons.
You've given me a tardigrade sized brain. Let's begin!
What do you choose? (rock:r, paper:p, scissors:s) or (save, load, exit)load
Please enter the path to load(i.e. rps_model.pt):rps_model_tardigrade.pt
Model loaded from  rps_model_tardigrade.pt
Model size is 2723 parameters spread out over 67 neurons.
You've given me a tardigrade sized brain. Let's begin!
What do you choose? (rock:r, paper:p, scissors:s) or (save, load, exit)r
torch.Size([1, 15]) torch.Size([1, 3]) torch.Size([1])
Loss is  0.9614773392677307
You lose!
The AI chose 1
Win ratio:  0.5975609756097561 Total games:  246
What do you choose? (rock:r, paper:p, scissors:s) or (save, load, exit)

```
