import os
import sys
import numpy as np
import json
import jsonlines
import collections
import argparse

# Define the split names
split_names = ['taskvars_train', 'taskvars_test_l2', 'taskvars_test_l3', 'taskvars_test_l4']

def main(args):
    # 若使用者未指定 output，則預設為 {result_dir}/evaluation_output.txt
    if not args.output:
        args.output = os.path.join(args.result_dir, "evaluation_output.txt")
    
    # Open output file for writing (UTF-8)
    fout = open(args.output, 'w', encoding='utf-8')
    
    def log_print(s=""):
        """Print to screen and also write to the output file."""
        print(s)
        fout.write(s + "\n")
    
    # Dictionary to store the results for each task
    results = collections.defaultdict(list)

    # Print seeds information
    log_print("Seed list: " + ", ".join(map(str, args.seeds)))
    log_print("")

    # Read results for each seed
    for seed in args.seeds:
        log_print(f'Loading seed {seed} ...')
        result_file = os.path.join(args.result_dir, f'seed{seed}', 'results.jsonl')
        if os.path.exists(result_file):
            with jsonlines.open(result_file, 'r') as f:
                for item in f:
                    # Ensure checkpoint is an int
                    if isinstance(item['checkpoint'], int):
                        res_ckpt = item['checkpoint']
                    else:
                        res_ckpt = int(os.path.basename(item['checkpoint']).split('_')[-1].split('.')[0])
                    # Only take the results from the specified checkpoint step
                    if res_ckpt == args.ckpt_step:
                        # Construct a task identifier as "task+variation"
                        taskvar = f"{item['task']}+{item['variation']}"
                        results[taskvar].append(item['sr'])
        else:
            log_print(f"{result_file} is missing")

    log_print("\nNote:")
    log_print("Each score represents the task success rate (multiplied by 100 to form a percentage).")
    log_print("The standard deviation indicates the variability between different seed results.\n")

    # Process each split's tasks
    for split_name in split_names:
        log_print(f"==== Split: {split_name} ====")
        # Load and sort the task list from the assets folder
        taskvars_file = os.path.join('assets', f'{split_name}.json')
        with open(taskvars_file, 'r', encoding='utf-8') as f:
            taskvars = json.load(f)
        taskvars.sort()

        # Calculate average and standard deviation (converted to percentage) for each task
        taskvars_sr = [np.mean(results[taskvar]) * 100 for taskvar in taskvars]
        taskvars_std = [np.std(results[taskvar]) * 100 for taskvar in taskvars]

        # Print header for the table
        header = f"{'Task':<40s} {'Avg Success Rate (%)':>20s} {'Std Dev (%)':>15s}"
        log_print(header)
        log_print("-" * len(header))
        # Print each task's results in a formatted row
        for task, avg_sr, std_sr in zip(taskvars, taskvars_sr, taskvars_std):
            log_print(f"{task:<40s} {avg_sr:20.2f} {std_sr:15.2f}")

        # Print overall averages
        overall_avg = np.mean(taskvars_sr)
        overall_std = np.mean(taskvars_std)
        log_print("-" * len(header))
        log_print(f"{'Overall Average':<40s} {overall_avg:20.2f}")
        log_print(f"{'Average Std Dev':<40s} {overall_std:20.2f}\n")

        # Calculate the average performance over seeds for this split
        num_seeds = min([len(results[taskvar]) for taskvar in taskvars])
        seed_results = [
            100 * np.mean([results[taskvar][i] for taskvar in taskvars])
            for i in range(min(len(args.seeds), num_seeds))
        ]
        log_print("Average success rate over seeds:")
        log_print("Mean: {:.2f} %, Std: {:.2f} %\n".format(np.mean(seed_results), np.std(seed_results)))
    
    fout.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate task results and display in a table format")
    parser.add_argument('result_dir', help="Path to the result directory")
    parser.add_argument('ckpt_step', type=int, help="Checkpoint step to filter results")
    parser.add_argument('--seeds', type=int, nargs='+', default=[200, 300, 400, 500, 600],
                        help="List of seeds to load")
    # 將 --output 預設設定為空字串，並在程式中根據 result_dir 做出預設值
    parser.add_argument('--output', type=str, default="",
                        help="Output file to write the results. Default: {result_dir}/evaluation_output.txt")
    args = parser.parse_args()
    
    main(args)
