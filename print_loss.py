import pickle
file = open("training_loss.txt", "rb")
losses = pickle.load(file)


for i in range(losses):
    print('The loss for epoch {i} is : ', losses[i])

