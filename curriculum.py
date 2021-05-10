import csv
import os
import tqdm


def load_csv(dataset, epoch):
    file_name = f"/home2/lgfm95/nas/darts/tempSave/curriculums/indices_{dataset}_{epoch}.csv"
    with open(file_name, "r") as fp:
        csvreader = csv.reader(fp)
        to_return = None
        for idx in csvreader: # trash method to get last + only elem
            to_return = idx
        return to_return


def load_all(dataset):
    epoch_dict = {}
    for file in tqdm.tqdm([f for f in os.listdir("/home2/lgfm95/nas/darts/tempSave/curriculums/") if f.endswith(".csv")]):
        print(file)
        elems = file[:-4].split("_")
        print(elems)
        epoch = elems[2]
        print(epoch)
        epoch_dict[epoch] = load_csv(dataset, epoch)
        print(epoch_dict[epoch])

    return epoch_dict


if __name__ == "__main__":
    load_all("mnist")