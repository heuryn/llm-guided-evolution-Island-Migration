from src.cfg.constants import *
import subprocess

# Taken from islands_wrapper.py
def submit_run(tempFile, text):
    """
    Bash script submission function for running a job on the cluster.
    Used for running each LLM's island.

    Parameters
    ----------
    tempFile : str
        Path to the temporary file where the bash script will be saved.
    text : str
        The content of the bash script to be executed.

    Returns
    -------
    job_id : str
        The job ID returned by the cluster after submitting the script.
        If submission fails, returns None.
    """
    with open(tempFile, 'w') as file:
        file.write(text)
    print(f"\t‣ Bash Script Saved to {tempFile}")
    job_id = None
    successful_sub_flag = False
    result = subprocess.run([RUN_COMMAND, tempFile], capture_output=True, text=True)
    if result.returncode == 0:
        print("\t‣ Script Submitted Successfully.\n\t‣ Output:", result.stdout.strip())
        successful_sub_flag = True
        job_id = result.stdout.split('job ')[-1].strip()
    else:
        print("\t‣ Failed to Submit script.\n\t‣ Error:", result.stderr.strip())
        successful_sub_flag = False
        job_id = None
    
    return job_id


if __name__ == "__main__":
    print(f"Starting inference servers for: {ISLAND_LLMS}")
    # initialize the inference servers for each island
    # using the temp island script sh file
    island_script = "src/island_temp_script.sh"
    for llm in ISLAND_LLMS:
        print(f"Starting inference server for {llm} stored at {LLM_ROOT_PATH + LLM_PATHS[llm]}")
        submit_run(island_script, LLM_INFERENCE_SERVER_TEMPLATE.format(PORT, LLM_ROOT_PATH + LLM_PATHS[llm]))