from warnings import filterwarnings

filterwarnings("ignore")


import os
import pickle
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.mixture import GaussianMixture

import dataloader_clothing1M as dataloader
from preset_parser import *

if __name__ == "__main__":
    args = parse_args("./presets.json")
    logs = open(os.path.join(args.checkpoint_path, "saved", "metrics.log"), "a")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Training
    def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader):
        net.train()
        net2.eval()  # fix one network and train the other

        unlabeled_train_iter = iter(unlabeled_trainloader)
        num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1
        for (
            batch_idx,
            (
                inputs_x,
                inputs_x2,
                inputs_x3,
                inputs_x4,
                labels_x,
                w_x,
            ),
        ) in enumerate(labeled_trainloader):
            try:
                inputs_u, inputs_u2, inputs_u3, inputs_u4 = unlabeled_train_iter.next()
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                inputs_u, inputs_u2, inputs_u3, inputs_u4 = unlabeled_train_iter.next()
            batch_size = inputs_x.size(0)

            # Transform label to one-hot
            labels_x = torch.zeros(batch_size, args.num_class).scatter_(
                1, labels_x.view(-1, 1), 1
            )
            w_x = w_x.view(-1, 1).type(torch.FloatTensor)

            inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x = (
                inputs_x.cuda(),
                inputs_x2.cuda(),
                inputs_x3.cuda(),
                inputs_x4.cuda(),
                labels_x.cuda(),
                w_x.cuda(),
            )
            inputs_u, inputs_u2, inputs_u3, inputs_u4 = (
                inputs_u.cuda(),
                inputs_u2.cuda(),
                inputs_u3.cuda(),
                inputs_u4.cuda(),
            )

            with torch.no_grad():
                # label co-guessing of unlabeled samples
                outputs_u_1 = net(inputs_u3)
                outputs_u_2 = net(inputs_u4)
                outputs_u_3 = net2(inputs_u3)
                outputs_u_4 = net2(inputs_u4)

                pu = (
                    torch.softmax(outputs_u_1, dim=1)
                    + torch.softmax(outputs_u_2, dim=1)
                    + torch.softmax(outputs_u_3, dim=1)
                    + torch.softmax(outputs_u_4, dim=1)
                ) / 4
                ptu = pu ** (1 / args.T)  # temparature sharpening

                targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
                targets_u = targets_u.detach()

                # label refinement of labeled samples
                outputs_x_1 = net(inputs_x3)
                outputs_x_2 = net(inputs_x4)

                px = (
                    torch.softmax(outputs_x_1, dim=1)
                    + torch.softmax(outputs_x_2, dim=1)
                ) / 2
                px = w_x * labels_x + (1 - w_x) * px
                ptx = px ** (1 / args.T)  # temparature sharpening

                targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
                targets_x = targets_x.detach()

            # mixmatch
            l = np.random.beta(args.alpha, args.alpha)
            l = max(l, 1 - l)

            all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
            all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]

            mixed_input = (
                l * input_a[: batch_size * 2] + (1 - l) * input_b[: batch_size * 2]
            )
            mixed_target = (
                l * target_a[: batch_size * 2] + (1 - l) * target_b[: batch_size * 2]
            )

            logits = net(mixed_input)

            Lx = -torch.mean(
                torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1)
            )

            # regularization
            prior = torch.ones(args.num_class) / args.num_class
            prior = prior.cuda()
            pred_mean = torch.softmax(logits, dim=1).mean(0)
            penalty = torch.sum(prior * torch.log(prior / pred_mean))

            loss = Lx + penalty

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sys.stdout.write("\r")
            sys.stdout.write(
                "Clothing1M | Epoch [%3d/%3d] Iter[%3d/%3d]\t  Labeled loss: %.4f "
                % (epoch, args.num_epochs - 1, batch_idx + 1, num_iter, Lx.item())
            )
            sys.stdout.flush()

        sys.stdout.write("\r")
        sys.stdout.flush()

    def warmup(net, optimizer, dataloader):
        net.train()
        for batch_idx, (inputs, labels, path) in enumerate(dataloader):
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = CEloss(outputs, labels)

            penalty = conf_penalty(outputs)
            L = loss + penalty
            L.backward()
            optimizer.step()

            sys.stdout.write("\r")
            sys.stdout.write(
                "|Warm-up: Iter[%3d/%3d]\t CE-loss: %.4f  Conf-Penalty: %.4f"
                % (batch_idx + 1, args.num_batches, loss.item(), penalty.item())
            )
            sys.stdout.flush()

    def val(net, val_loader, k):
        net.eval()
        all_targets = []
        all_predicted = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = net(inputs)
                _, predicted = torch.max(outputs, 1)

                all_targets += targets.tolist()
                all_predicted += predicted.tolist()
        accuracy = accuracy_score(all_targets, all_predicted)
        precision = precision_score(all_targets, all_predicted, average="weighted")
        recall = recall_score(all_targets, all_predicted, average="weighted")
        f1 = f1_score(all_targets, all_predicted, average="weighted")
        print("\n| Validation\t Net%d  Acc: %.2f%%" % (k, accuracy * 100))
        if accuracy > best_acc[k - 1]:
            best_acc[k - 1] = accuracy
            print("| Saving Best Net%d ..." % k)
            save_point = os.path.join(
                args.checkpoint_path, "saved", args.preset + ".net%d.pth.tar" % (k)
            )
            torch.save(net.state_dict(), save_point)
        return accuracy, precision, recall, f1

    def test(net1, net2, test_loader):
        net1.eval()
        net2.eval()
        correct = 0
        total = 0
        all_targets = []
        all_predicted = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs1 = net1(inputs)
                outputs2 = net2(inputs)
                outputs = outputs1 + outputs2
                _, predicted = torch.max(outputs, 1)

                all_targets += targets.tolist()
                all_predicted += predicted.tolist()

        accuracy = accuracy_score(all_targets, all_predicted)
        precision = precision_score(all_targets, all_predicted, average="weighted")
        recall = recall_score(all_targets, all_predicted, average="weighted")
        f1 = f1_score(all_targets, all_predicted, average="weighted")
        results = (
            "Final Metrics, Accuracy: %.3f, Precision: %.3f, Recall: %.3f, F1: %.3f"
            % (
                accuracy * 100,
                precision * 100,
                recall * 100,
                f1 * 100,
            )
        )
        print("\n" + results + "\n")
        logs.write(results + "\n")
        logs.flush()

    def eval_train(epoch, model):
        model.eval()
        num_samples = args.num_batches * args.batch_size
        losses = torch.zeros(num_samples)
        paths = []
        n = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets, path) in enumerate(eval_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                loss = CE(outputs, targets)
                for b in range(inputs.size(0)):
                    losses[n] = loss[b]
                    paths.append(path[b])
                    n += 1
                sys.stdout.write("\r")
                sys.stdout.write("| Evaluating loss Iter %3d\t" % (batch_idx))
                sys.stdout.flush()

        losses = (losses - losses.min()) / (losses.max() - losses.min())
        losses = losses.reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, max_iter=10, reg_covar=5e-4, tol=1e-2)
        gmm.fit(losses)
        prob = gmm.predict_proba(losses)
        prob = prob[:, gmm.means_.argmin()]
        return prob, paths

    class NegEntropy(object):
        def __call__(self, outputs):
            probs = torch.softmax(outputs, dim=1)
            return torch.mean(torch.sum(probs.log() * probs, dim=1))

    def create_model(devices=[0]):
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(2048, args.num_class)
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=devices).cuda()
        return model

    loader = dataloader.clothing_dataloader(
        root=args.data_path,
        batch_size=args.batch_size,
        warmup_batch_size=args.warmup_batch_size,
        num_workers=args.num_workers,
        num_batches=args.num_batches,
        augmentation_strategy=args,
    )

    print("| Building net")
    devices = range(torch.cuda.device_count())
    net1 = create_model(devices)
    net2 = create_model(devices)
    cudnn.benchmark = True

    optimizer1 = optim.SGD(
        net1.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-3
    )
    optimizer2 = optim.SGD(
        net2.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-3
    )
    best_acc = [0, 0]

    if args.pretrained_path != "":
        with open(args.pretrained_path + f"/saved/{args.preset}.pkl", "rb") as p:
            unpickled = pickle.load(p)
        net1.load_state_dict(unpickled["net1"])
        net2.load_state_dict(unpickled["net2"])
        optimizer1.load_state_dict(unpickled["optimizer1"])
        optimizer2.load_state_dict(unpickled["optimizer2"])
        best_acc = unpickled["best_acc"]
        epoch = unpickled["epoch"] + 1
    else:
        epoch = 0

    CE = nn.CrossEntropyLoss(reduction="none")
    CEloss = nn.CrossEntropyLoss()
    conf_penalty = NegEntropy()

    while epoch < args.num_epochs:
        lr = args.learning_rate
        if epoch >= args.lr_switch_epoch:
            lr /= 10
        for param_group in optimizer1.param_groups:
            param_group["lr"] = lr
        for param_group in optimizer2.param_groups:
            param_group["lr"] = lr

        if epoch < args.warm_up:  # warm up
            train_loader = loader.run("warmup")
            print("Warmup Net1")
            warmup(net1, optimizer1, train_loader)
            train_loader = loader.run("warmup")
            print("\nWarmup Net2")
            warmup(net2, optimizer2, train_loader)
            size_l1, size_u1, size_l2, size_u2 = (
                len(train_loader.dataset),
                0,
                len(train_loader.dataset),
                0,
            )
        else:
            sys.stdout.flush()
            print("\n==== net 1 evaluate next epoch training data loss ====")
            eval_loader = loader.run(
                "eval_train"
            )  # evaluate training data loss for next epoch
            prob1, paths1 = eval_train(epoch, net1)
            print("\n==== net 2 evaluate next epoch training data loss ====")
            eval_loader = loader.run("eval_train")
            prob2, paths2 = eval_train(epoch, net2)

            pred1 = prob1 > args.p_threshold  # divide dataset
            pred2 = prob2 > args.p_threshold

            print("\n\nTrain Net1")
            labeled_trainloader, unlabeled_trainloader = loader.run(
                "train", pred2, prob2, paths=paths2
            )  # co-divide
            size_l1, size_u1 = (
                len(labeled_trainloader.dataset),
                len(unlabeled_trainloader.dataset),
            )
            train(
                epoch,
                net1,
                net2,
                optimizer1,
                labeled_trainloader,
                unlabeled_trainloader,
            )  # train net1
            print("\nTrain Net2")
            labeled_trainloader, unlabeled_trainloader = loader.run(
                "train", pred1, prob1, paths=paths1
            )  # co-divide
            size_l2, size_u2 = (
                len(labeled_trainloader.dataset),
                len(unlabeled_trainloader.dataset),
            )
            train(
                epoch,
                net2,
                net1,
                optimizer2,
                labeled_trainloader,
                unlabeled_trainloader,
            )  # train net2

        val_loader = loader.run("val")  # validation
        acc1, prec1, rec1, f1_1 = val(net1, val_loader, 1)
        acc2, prec2, rec2, f1_2 = val(net2, val_loader, 2)
        results = "Test Epoch: %d, Accuracy: %.3f & %.3f, Precision: %.3f & %.3f, Recall: %.3f & %.3f, F1: %.3f & %.3f, L_1: %d, U_1: %d, L_2: %d, U_2: %d" % (
            epoch,
            acc1 * 100,
            acc2 * 100,
            prec1 * 100,
            prec2 * 100,
            rec1 * 100,
            rec2 * 100,
            f1_1 * 100,
            f1_2 * 100,
            size_l1,
            size_u1,
            size_l2,
            size_u2,
        )
        print("\n" + results + "\n")
        logs.write(results + "\n")
        logs.flush()
        if (epoch + 1) % args.save_every == 0 or epoch == args.warm_up - 1:
            data_dict = {
                "epoch": epoch,
                "net1": net1.state_dict(),
                "net2": net2.state_dict(),
                "optimizer1": optimizer1.state_dict(),
                "optimizer2": optimizer2.state_dict(),
                "best_acc": best_acc,
            }
            checkpoint_model = open(
                os.path.join(args.checkpoint_path, "all", f"model_{epoch}.pkl"), "wb"
            )
            pickle.dump(data_dict, checkpoint_model)
            saved_model = open(
                os.path.join(args.checkpoint_path, "saved", f"{args.preset}.pkl"), "wb"
            )
            pickle.dump(data_dict, saved_model)
        epoch += 1

    test_loader = loader.run("test")
    net1.load_state_dict(
        torch.load(
            os.path.join(args.checkpoint_path, "saved", args.preset + ".net1.pth.tar")
        )
    )
    net2.load_state_dict(
        torch.load(
            os.path.join(args.checkpoint_path, "saved", args.preset + ".net2.pth.tar")
        )
    )
    test(net1, net2, test_loader)
