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

To save/load a previous model, just use the prompts after the above is running. Type `save` or `load` and it will ask you for the file path. If it's in the same file path as the above run script you created, just type the file name only. A slightly pretrained model the size of a tardigrade's brain is inside of the rpsskynet folder on the Github repo page. 

If you are starting to train a model from scratch, be sure to play at least 200 games to initialize it's parameters. Then it should start performing better. 

This was developed for fun and can be used as a training tool for new ML learners. The only file in the repo is under 130 lines of code.
