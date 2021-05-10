import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
def main():
    training_loss = []
    translation_error = []
    with open('modelnet10_0509.txt') as f:
        lines = f.readlines()
        # print(lines, '\n')
        for line in lines:
            if "Loss" in line:
                if "Batch" in line:
                    continue
                # print(line)
                loss = float(line.split(": ")[1].rstrip("\n"))
                training_loss.append(loss)
            if "translation error" in line:
                trans_err = float(line.split(": ")[1].rstrip("\n"))
                translation_error.append(trans_err)
    
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(training_loss, color='blue', label="training")
    axs[0].set_title('Training Loss')
    axs[0].set_ylabel('loss')
    axs[1].plot(translation_error, color='blue', label="training")
    axs[1].set_title('Translation Error')
    axs[1].set_ylabel('Translation Error')
    plt.show()
if __name__ == "__main__":
    main()
