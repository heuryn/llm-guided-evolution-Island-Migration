import os
import sys
import subprocess

# ======== CONFIGURE THESE ========
# Root folder containing your 6 factor folders (2x3)
BASE_DIR = r"/Users/jackwarren/Documents/VIP-AAD/reduced_tests"

# Path to your metric script
# (assuming it's a Python script; see comment below if it's not)
METRIC_SCRIPT = r"/Users/jackwarren/Documents/VIP-AAD/reduced_tests/fitness_diversity_quantifier.py"
# =================================


def main():
    # Folder where all logs will be stored
    log_dir = os.path.join(BASE_DIR, "metric_logs")
    os.makedirs(log_dir, exist_ok=True)

    # Loop over the 6 factor folders (cells)
    for cell_name in sorted(os.listdir(BASE_DIR)):
        cell_path = os.path.join(BASE_DIR, cell_name)
        if not os.path.isdir(cell_path):
            continue  # skip files like logs, etc.

        # Loop over the 6 sample folders inside each cell
        for sample_name in sorted(os.listdir(cell_path)):
            sample_path = os.path.join(cell_path, sample_name)
            if not os.path.isdir(sample_path):
                continue

            # Name of the log file for this sample
            log_filename = f"{cell_name}__{sample_name}.log"
            log_path = os.path.join(log_dir, log_filename)

            #print(f"Running metrics for: {sample_path}")
            #print(f"Cell Path: {sample_name}")
            actual_path = os.path.join(cell_name, sample_name)
            print(f"Running metrics for: {actual_path}")

            # Command to run your metric script
            # If your script is Python:
            
            cmd = [sys.executable, METRIC_SCRIPT, actual_path, "--include-hist", "--front-only"]

            # If your metric thing is a shell command instead, e.g. "my_metric_tool":
            # cmd = ["my_metric_tool", sample_path]

            result = subprocess.run(
                cmd,
                cwd=BASE_DIR,         
                capture_output=True,
                text=True
            )
            
            # Save stdout, stderr, and return code to log file
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Working directory: {sample_path}\n")
                f.write(f"Return code: {result.returncode}\n\n")

                f.write("==== STDOUT ====\n")
                f.write(result.stdout or "")
                f.write("\n\n==== STDERR ====\n")
                f.write(result.stderr or "")
            

    print("\nDone. Logs saved in:", log_dir)


if __name__ == "__main__":
    main()
