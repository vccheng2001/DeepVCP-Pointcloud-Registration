import numpy as np
import matplotlib.pyplot as plt

def main():
    training_loss = []
    with open('nohup_local.out') as f:
        lines = f.readlines()
        # print(lines, '\n')
        for line in lines:
            if "Loss" in line:
                if "Batch" in line:
                    continue
                # print(line)
                loss = float(line.split(": ")[1].rstrip("\n"))
                training_loss.append(loss)
    
    fig, ax1 = plt.subplots(1, 1)

    ax1.plot(training_loss, color='blue', label="training")
    ax1.set_title('Training Loss')
    ax1.set_ylabel('loss')
    plt.show()


if __name__ == "__main__":
    main()