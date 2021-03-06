import csv
import os
import tqdm
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import random


def load_csv(dataset, epoch):
    file_name = f"/home2/lgfm95/nas/darts/tempSave/curriculums/indices_{dataset}_{epoch}.csv"
    with open(file_name, "r") as fp:
        elems = fp.readlines()[0]
        elems = elems.split(" ")
        return [int(elem) for elem in elems]


def load_all(dataset):
    epoch_dict = {}
    for file in tqdm.tqdm([f for f in os.listdir("/home2/lgfm95/nas/darts/tempSave/curriculums/") if f.endswith(".csv") and dataset in f]):
        # print(file)
        elems = file[:-4].split("_")
        # print(elems)
        epoch = int(elems[2])
        # print(epoch)
        epoch_dict[epoch] = load_csv(dataset, epoch)
        # print(epoch_dict[epoch])

    return epoch_dict


def get_split(dataset):
    n_train = len(dataset)
    split = n_train * 0.8
    remainder = split % 8
    return int(split - remainder)


class Curriculum_loader():
    def __init__(self, dataset, val):
        self.dataset = dataset
        self.epoch_dict = load_all(dataset)
        self.update_epochs = self.epoch_dict.keys()
        tensor_transform = transforms.Compose([transforms.ToTensor()])

        if dataset == "mnist":
            print ("using mnist in subloader")
            train_path = "/home2/lgfm95/mnist/"
            self.full_set = list(datasets.MNIST(train_path, train=True, transform=tensor_transform, target_transform=None, download=False))
        elif dataset == "cifar10":
            print("using cifar10 in subloader")
            self.full_set = list(datasets.CIFAR10("/home2/lgfm95/cifar10/", train=True, transform=tensor_transform, target_transform=None,download=False))
        else:
            raise AttributeError("not mnist or cifar10")
        split = get_split(self.full_set)
        indices = list(range(len(self.full_set)))
        random.seed(1337)  # note must use same random seed as subloader (and thus process same images as subloader)
        random.shuffle(indices)
        self.train_indices, self.val_indices = indices[:split], indices[split:]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.train_indices)
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.val_indices)
        if val:
            self.split_loader = torch.utils.data.DataLoader(self.full_set,
                                                            batch_size=1,
                                                            sampler=valid_sampler,
                                                            num_workers=0,
                                                            pin_memory=False)
        else:
            self.split_loader = torch.utils.data.DataLoader(self.full_set,
                                                            batch_size=1,
                                                            sampler=train_sampler,
                                                            num_workers=0,
                                                            pin_memory=False)

        self.data, self.fine = zip(*list(self.split_loader))

        self.generate_cur_set(0)
        self.len = len(self.cur_set)

    def __len__(self):
        if self.dataset == "mnist":
            assert self.len == 100
        elif self.dataset == "cifar10":
            assert self.len == 1000
        return self.len

    def __getitem__(self, item):
        return self.cur_set[item], self.fine_set[item]

    def generate_cur_set(self, epoch):
        # self.cur_set = []
        # self.fine_set = []
        # for idx in self.epoch_dict[epoch]:
        #     try:
        #         self.cur_set.append(self.data[int(idx)])
        #         self.fine_set.append(self.fine[int(idx)])
        #     except ValueError:
        #         print(f"guilty_{idx}_")
        #         continue
        self.cur_set = [self.data[idx] for idx in self.epoch_dict[epoch]]
        self.fine_set = [self.fine[idx] for idx in self.epoch_dict[epoch]]


if __name__ == "__main__":
    load_all("mnist")
