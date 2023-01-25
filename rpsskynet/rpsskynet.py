import torch
import torch.nn as nn
import time

rps_dict={'r':0, 'p':1,'s':2}
def check_winner(user_choice, ai_choice, winloss): # get winner
    # model trained to predict user number, shift model choice to the right one
    if ai_choice==2: ai_choice*=0
    else: ai_choice+=1
    if user_choice == ai_choice:
        print("It's a tie!")
        winloss += 0.5
    elif user_choice.item() == 0 and ai_choice.item() == 2:
        print("You win!")
        winloss+=1
    elif user_choice.item() == 2 and ai_choice.item() == 1:
        print("You win!")
        winloss += 1
    elif user_choice.item() == 1 and ai_choice.item() == 0:
        print("You win!")
        winloss += 1
    else:
        print("You lose!")
    print("The AI chose", ai_choice.item())
    return winloss

def get_neuron_animal_size(num_params): # animal neuron count comparison
    animbrain={200:"tardigrade", 300: "roundworm", 5600: "jellyfish", 100000: "fruit fly", 250000: "ant",
        960000: "honeybee", 1771000:"gecko", 4300000: "guppy", 8500000: "tortoise", 16000000: "frog", 71000000: "household mouse",
    310000000:"pigeon", 500000000: "octopus", 760000000: "household cat", 2253000000: "dog", 86000000000: "human", 257000000000: "elephant"}
    res_key, res_anim = min(animbrain.items(), key=lambda x: abs(num_params - x[0]))
    print("You've given me a", res_anim, "sized brain. Let's begin!")

# Network definition
class RPSModel(torch.nn.Module):
    def __init__(self, game_memory:int, hidden_size:int, mid_layers:int):
        super().__init__()
        self.hidden_size=hidden_size
        self.mid_layers=mid_layers
        self.in_features=3*game_memory
        self.layer_1=torch.nn.Linear(self.in_features,hidden_size)
        self.relu=nn.LeakyReLU()
        self.layer_mid=torch.nn.ModuleList([])
        for i in range(mid_layers):
            self.layer_mid.append(nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LeakyReLU()))
        self.layer_out=torch.nn.Linear(hidden_size,3)

    def forward(self, x):
        x=self.layer_1(x)
        x=self.relu(x)
        for layer in self.layer_mid:
            x=layer(x)
        x=self.layer_out(x)
        return x

def model_startup(model): #initial text script after model is loaded
    num_params = sum(p.numel() for p in model.parameters())
    num_neurons = int(model.hidden_size) * int(model.mid_layers) + (model.in_features)+3
    print("Model size is", num_params, "parameters spread out over", num_neurons, "neurons.")
    time.sleep(1)
    get_neuron_animal_size(num_neurons)
    time.sleep(1)

def rps_skynet(game_memory = 5, batchsize = 128):
    hiddensize=input("Welcome to Rock, Paper, Scissors Skynet! Please enter my hidden_size(i.e. 32): ")
    mid_layers=input("Please enter the number of mid_layers(Hint: 2; under 8 to avoid vanishing gradients): ")
    model=RPSModel(game_memory, int(hiddensize), int(mid_layers))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model_startup(model)
    game_stack=torch.empty((0,3*game_memory))
    old_games=torch.nn.functional.one_hot(torch.randint(0, 2, (1,game_memory)),3).float().view(1,-1)
    game_stack=torch.cat([game_stack, old_games])
    model_output=model(game_stack)
    n_games=winloss=0
    targets=torch.empty(0, dtype=torch.long)
    while True:
        user_input1=input('What do you choose? (rock:r, paper:p, scissors:s) or (save, load, exit): ')
        if user_input1 in rps_dict:
            user_input=torch.tensor(rps_dict[user_input1]).view(-1)
            targets=torch.cat([user_input, targets])
            targets=targets[:batchsize]
        else:
            if user_input1[:4]=="save":
                path=input('Please enter the path to save(i.e. rps_model.pt): ')
                torch.save({
                    'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                    'n_games' : n_games, 'winloss' : winloss, 'hiddensize':hiddensize, 'midlayers':mid_layers,
                    'old_games':old_games, 'game_memory':game_memory
                }, path)
                print("Model saved to ", path)
                continue
            elif user_input1[:4]=="load":
                path=input('Please enter the path to load(i.e. rps_model.pt): ')
                checkpoint=torch.load(path)
                mid_layers=checkpoint['midlayers']
                hiddensize=checkpoint['hiddensize']
                game_memory=checkpoint['game_memory']
                del model
                model=RPSModel(int(game_memory), int(hiddensize), int(mid_layers))
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                n_games=checkpoint['n_games']
                winloss=checkpoint['winloss']
                print("Model loaded from ", path)
                old_games=checkpoint['old_games']
                model_startup(model)
                continue
            elif user_input1[:4]=="exit":
                print("Farewell and better luck next time!")
                break
            else:
                print("Your entry was not clear.")
                continue
        print(game_stack.size(), model_output.size(), targets.size())
        loss=criterion(model_output, targets)
        loss.backward()
        optimizer.step()
        print("Loss is ", loss.detach().item())
        winloss=check_winner(user_input, torch.argmax(model_output[:1,:].detach()), winloss)
        n_games+=1
        print("Win ratio: ", winloss/n_games, "Total games: ", n_games)
        optimizer.zero_grad()
        old_games=torch.cat([old_games, torch.nn.functional.one_hot(user_input.detach(),3).float()],dim=1)[:,3:]
        game_stack=torch.cat([old_games, game_stack])
        game_stack=game_stack[:batchsize,:]
        model_output = model(game_stack)
