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
    
    # cut the length of the lists into multiples of 10
    new_len = (len(training_loss) // 10) * 10
    translation_error = translation_error[:new_len]
    training_loss = training_loss[:new_len]

    # calculate the average loss and translation error every 10 batches
    translation_error_arr = np.asarray(translation_error)
    translation_error_avg_batch = np.mean(translation_error_arr.reshape(-1, 10), axis = 1)
    training_loss_arr = np.asarray(training_loss)
    training_loss_avg_batch = np.mean(training_loss_arr.reshape(-1, 10), axis = 1)

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(training_loss_avg_batch, color='blue', label="training")
    axs[0].set_title('Training Loss')
    axs[0].set_ylabel('loss')
    axs[1].plot(translation_error_avg_batch, color='blue', label="training")
    axs[1].set_title('Translation Error')
    axs[1].set_ylabel('Translation Error')
    plt.show()
if __name__ == "__main__":
    main()
