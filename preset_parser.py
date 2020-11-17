import argparse
import json
import os


def parse_args(file):
    parser = argparse.ArgumentParser(description="PyTorch CIFAR Training")
    parser.add_argument("--preset", required=True, type=str)
    parser.add_argument("--machine", required=False, type=str, default="")
    cmdline_args = parser.parse_args()

    with open(file, "r") as f:
        jsonFile = json.load(f)

    class dotdict(dict):
        def __getattr__(self, name):
            return self[name]

        def __setattr__(self, name, value):
            self[name] = value

    args = dotdict()
    args.update(jsonFile)
    if "machines" in jsonFile and cmdline_args.machine in jsonFile["machines"]:
        args.update(jsonFile["machines"][cmdline_args.machine])
    if "configs" in args:
        del args["configs"]
        jsonFile = jsonFile["configs"]

    args.preset = cmdline_args.preset
    subpresets = cmdline_args.preset.split(".")
    for subp in subpresets:
        jsonFile = jsonFile[subp]
        args.update(jsonFile)
        if "configs" in args:
            del args["configs"]
        if "machines" in args:
            if cmdline_args.machine != "":
                args.update(jsonFile["machines"][cmdline_args.machine])
            del args["machines"]
        if "configs" in jsonFile:
            jsonFile = jsonFile["configs"]

    args.checkpoint_path += "/" + "/".join(args.preset.split("."))
    args.pretrained_path = args.checkpoint_path

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    all_folder = os.path.join(args.checkpoint_path, "all")
    if not os.path.exists(all_folder):
        os.mkdir(all_folder)
    saved_folder = os.path.join(args.checkpoint_path, "saved")
    if not os.path.exists(saved_folder):
        os.mkdir(saved_folder)
    if not os.path.exists(args.pretrained_path + f"/saved/{args.preset}.pth.tar"):
        # if os.path.exists(args.pretrained_path + f"/saved/metrics.log"):
        #     raise AssertionError("Training log already exists!")
        args.pretrained_path = ""

    with open(args.checkpoint_path + "/saved/info.json", "w") as f:
        json.dump(args, f, indent=4, sort_keys=True)

    return args
