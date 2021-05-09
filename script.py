import pickle
file = open("training_loss.txt", "rb")
data = pickle.load(file)


cnt = 0
for item in data:
    print('The data ', cnt, ' is : ', item)
    cnt += 1

