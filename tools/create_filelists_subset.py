import argparse
import json
import os


def subset_filelists(repeat_times, subset_speakers, input_list, output_list):
    with open(input_list) as fp_in, open(output_list, 'w') as fp_out:
        lines = fp_in.readlines()
        for s, n in zip(subset_speakers, repeat_times):
            for line in lines:
                if line.rsplit("/", 2)[1] == s:
                    for _ in range(n):
                        fp_out.write(line)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, help="The subset speaker list")
    parser.add_argument("--repeat", type=str, help="The num of repeat times for each subset speaker")
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    repeat_times = [int(i) for i in args.repeat.split(",")]
    subset_speakers = args.subset.split(",")

    with open(os.path.join(args.input_dir, "config.json")) as f:
        config = json.load(f)
    
    training_files = config["data"]["training_files"]
    validation_files = config["data"]["validation_files"]

    config["data"]["training_files"] = os.path.join(args.output_dir, "train.txt")
    config["data"]["validation_files"] = os.path.join(args.output_dir, "val.txt")

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    subset_filelists(repeat_times, subset_speakers, training_files, os.path.join(args.output_dir, "train.txt"))
    subset_filelists([1]*len(subset_speakers), subset_speakers, validation_files, os.path.join(args.output_dir, "val.txt"))

if __name__ == '__main__':
    main()