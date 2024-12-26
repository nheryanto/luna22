from time import time
from datetime import datetime
def print_to_log_file(log_file, *args, also_print_to_console=True, add_timestamp=True):
    timestamp = time()
    dt_object = datetime.fromtimestamp(timestamp)

    if add_timestamp:
        args = (f"{dt_object}:", *args)

    with open(log_file, 'a+') as f:
        for a in args:
            f.write(str(a))
            f.write(" ")
        f.write("\n")

    if also_print_to_console:
        print(*args)
        
def confusion_matrix(preds, targets):
    TP = TN = FP = FN = 0

    for true, pred in zip(targets, preds):
        if true == 1 and pred == 1:
            TP += 1  # True Positive
        elif true == 0 and pred == 0:
            TN += 1  # True Negative
        elif true == 0 and pred == 1:
            FP += 1  # False Positive
        elif true == 1 and pred == 0:
            FN += 1  # False Negative

    return TP, TN, FP, FN
        
from os.path import join
def plot_epochs(train_loss, val_loss, val_dice, epoch_time, output_dir):
    fig = plt.figure(figsize=(8,10), dpi=1200)
    with plt.style.context("seaborn-v0_8-whitegrid"):
        ax1 = fig.add_subplot(211)
        ax1.plot(train_loss, linestyle="dashed", color="blue", label="train_loss")
        ax1.plot(val_loss, linestyle="dotted", color="red", label="val_loss")
        ax1.plot(val_dice, linestyle="solid", color="green", label="val_dice")
        ax1.set_ylabel("loss", fontsize=12, labelpad=10)
        ax1.set_xlabel("epoch", fontsize=12, labelpad=10)
        ax1.tick_params(axis="both", which="major")
        ax1.legend(loc="best")
        secax1 = ax1.secondary_yaxis("right")
        secax1.set_ylabel("dice score", fontsize=12, labelpad=10)
        secax1.tick_params(axis="both", which="major")

        ax2 = fig.add_subplot(212)
        ax2.plot(epoch_time, linestyle="dashed", color="blue", label="epoch_time")
        ax2.set_ylabel('time [s]', fontsize=12, labelpad=10)
        ax2.set_xlabel('epoch', fontsize=12, labelpad=10)
        ax2.tick_params(axis='both', which='major')
        ax2.legend(loc='best')
        
        plt.savefig(join(output_dir, "progress.png"))
        plt.close()