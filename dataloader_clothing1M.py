from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import torch
from autoaugment import CIFAR10Policy, ImageNetPolicy


transform_weak_c1m_c10_compose = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
    ]
)


def transform_weak_c1m(x):
    return transform_weak_c1m_c10_compose(x)


transform_strong_c1m_c10_compose = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
    ]
)


def transform_strong_c1m_c10(x):
    return transform_strong_c1m_c10_compose(x)


transform_strong_c1m_in_compose = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        ImageNetPolicy(),
        transforms.ToTensor(),
        transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
    ]
)


def transform_strong_c1m_in(x):
    return transform_strong_c1m_in_compose(x)


class clothing_dataset(Dataset):
    def __init__(
        self,
        root,
        transform,
        mode,
        num_samples=0,
        pred=[],
        probability=[],
        paths=[],
        num_class=14,
    ):

        self.root = root
        self.transform = transform
        self.mode = mode
        self.train_labels = {}
        self.test_labels = {}
        self.val_labels = {}

        with open("%s/noisy_label_kv.txt" % self.root, "r") as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = "%s/" % self.root + entry[0][7:]
                self.train_labels[img_path] = int(entry[1])
        with open("%s/clean_label_kv.txt" % self.root, "r") as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = "%s/" % self.root + entry[0][7:]
                self.test_labels[img_path] = int(entry[1])

        if mode == "all":
            train_imgs = []
            with open("%s/noisy_train_key_list.txt" % self.root, "r") as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = "%s/" % self.root + l[7:]
                    train_imgs.append(img_path)
            random.shuffle(train_imgs)
            class_num = torch.zeros(num_class)
            self.train_imgs = []
            for impath in train_imgs:
                label = self.train_labels[impath]
                if (
                    class_num[label] < (num_samples / 14)
                    and len(self.train_imgs) < num_samples
                ):
                    self.train_imgs.append(impath)
                    class_num[label] += 1
            random.shuffle(self.train_imgs)
        elif self.mode == "labeled":
            train_imgs = paths
            pred_idx = pred.nonzero()[0]
            self.train_imgs = [train_imgs[i] for i in pred_idx]
            self.probability = [probability[i] for i in pred_idx]
            print("%s data has a size of %d" % (self.mode, len(self.train_imgs)))
        elif self.mode == "unlabeled":
            train_imgs = paths
            pred_idx = (1 - pred).nonzero()[0]
            self.train_imgs = [train_imgs[i] for i in pred_idx]
            self.probability = [probability[i] for i in pred_idx]
            print("%s data has a size of %d" % (self.mode, len(self.train_imgs)))

        elif mode == "test":
            self.test_imgs = []
            with open("%s/clean_test_key_list.txt" % self.root, "r") as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = "%s/" % self.root + l[7:]
                    self.test_imgs.append(img_path)
        elif mode == "val":
            self.val_imgs = []
            with open("%s/clean_val_key_list.txt" % self.root, "r") as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = "%s/" % self.root + l[7:]
                    self.val_imgs.append(img_path)

    def __getitem__(self, index):
        if self.mode == "labeled":
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            prob = self.probability[index]
            image = Image.open(img_path).convert("RGB")
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            if self.transform[2] == None:
                img3 = img1
                img4 = img2
            else:
                img3 = self.transform[2](image)
                img4 = self.transform[3](image)
            return img1, img2, img3, img4, target, prob
        elif self.mode == "unlabeled":
            img_path = self.train_imgs[index]
            image = Image.open(img_path).convert("RGB")
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            if self.transform[2] == None:
                img3 = img1
                img4 = img2
            else:
                img3 = self.transform[2](image)
                img4 = self.transform[3](image)
            return img1, img2, img3, img4
        elif self.mode == "all":
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            image = Image.open(img_path).convert("RGB")
            img = self.transform(image)
            return img, target, img_path
        elif self.mode == "test":
            img_path = self.test_imgs[index]
            target = self.test_labels[img_path]
            image = Image.open(img_path).convert("RGB")
            img = self.transform(image)
            return img, target
        elif self.mode == "val":
            img_path = self.val_imgs[index]
            target = self.test_labels[img_path]
            image = Image.open(img_path).convert("RGB")
            img = self.transform(image)
            return img, target

    def __len__(self):
        if self.mode == "test":
            return len(self.test_imgs)
        if self.mode == "val":
            return len(self.val_imgs)
        else:
            return len(self.train_imgs)


class clothing_dataloader:
    def __init__(
        self,
        root,
        batch_size,
        warmup_batch_size,
        num_batches,
        num_workers,
        augmentation_strategy={},
    ):
        self.batch_size = batch_size
        self.warmup_batch_size = warmup_batch_size
        self.num_workers = num_workers
        self.num_batches = num_batches
        self.root = root
        self.augmentation_strategy = augmentation_strategy

        self.transforms = {
            "warmup": globals()[augmentation_strategy.warmup_transform],
            "unlabeled": [None for i in range(4)],
            "labeled": [None for i in range(4)],
            "test": None,
        }
        # workaround so it works on both windows and linux
        for i in range(len(augmentation_strategy.unlabeled_transforms)):
            self.transforms["unlabeled"][i] = globals()[
                augmentation_strategy.unlabeled_transforms[i]
            ]
        for i in range(len(augmentation_strategy.labeled_transforms)):
            self.transforms["labeled"][i] = globals()[
                augmentation_strategy.labeled_transforms[i]
            ]
        self.transforms["test"] = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)
                ),
            ]
        )

    def run(self, mode, pred=[], prob=[], paths=[]):
        if mode == "warmup":
            warmup_dataset = clothing_dataset(
                self.root,
                transform=self.transforms["warmup"],
                mode="all",
                num_samples=self.num_batches * self.warmup_batch_size,
            )
            warmup_loader = DataLoader(
                dataset=warmup_dataset,
                batch_size=self.warmup_batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )
            return warmup_loader
        elif mode == "train":
            labeled_dataset = clothing_dataset(
                self.root,
                transform=self.transforms["labeled"],
                mode="labeled",
                pred=pred,
                probability=prob,
                paths=paths,
            )
            labeled_loader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )
            unlabeled_dataset = clothing_dataset(
                self.root,
                transform=self.transforms["unlabeled"],
                mode="unlabeled",
                pred=pred,
                probability=prob,
                paths=paths,
            )
            unlabeled_loader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )
            return labeled_loader, unlabeled_loader
        elif mode == "eval_train":
            eval_dataset = clothing_dataset(
                self.root,
                transform=self.transforms["test"],
                # ^- this is a small mistake in our implementation!
                # augmentations for eval_train should be weak, not none.
                # although this has a neglibible effect on performance,
                # we feel it is important to note for future readers of this code.
                # see: https://github.com/KentoNishi/Augmentation-for-LNL/issues/4
                mode="all",
                num_samples=self.num_batches * self.batch_size,
            )
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            return eval_loader
        elif mode == "test":
            test_dataset = clothing_dataset(
                self.root, transform=self.transforms["test"], mode="test"
            )
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            return test_loader
        elif mode == "val":
            val_dataset = clothing_dataset(
                self.root, transform=self.transforms["test"], mode="val"
            )
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            return val_loader
