import os
import argparse
from deap import base, creator, tools
from deap.tools import HallOfFame
import islandIntegration
import islands
import subprocess
import time
from src.cfg.constants import *

def submit_run(tempFile, text):
    with open(tempFile, 'w') as file:
        file.write(text)
    print(f"\t‣ Bash Script Saved to {tempFile}")
    job_id = None
    successful_sub_flag = False
    result = subprocess.run(["sbatch", tempFile], capture_output=True, text=True)
    if result.returncode == 0:
        print("\t‣ Script Submitted Successfully.\n\t‣ Output:", result.stdout.strip())
        successful_sub_flag = True
        job_id = result.stdout.split('job ')[-1].strip()
    else:
        print("\t‣ Failed to Submit script.\n\t‣ Error:", result.stderr.strip())
        successful_sub_flag = False
        job_id = None
    
    return job_id
    
def check_contents_for_error(contents):
    """
    Checks the output of a job for any signs of error.

    Parameters:
    contents (str): output of job to check for error

    Returns:
    bool: True if job completed successfully, False if error, None if neither.  
    """

    # Check for error indicators in the file
    if "traceback" in contents.lower() or "slurmstepd: error" in contents.lower():
        print("\t☠ Error Found in LLM Job Output.", flush=True)
        return False
    elif "finished one generation" in contents.lower():
        print("\t☑ LLM Job Completed Successfully.", flush=True)
        return True
    else:
        return None

def check4job_completion(job_id, local_output=None, check_interval=60, timeout=3600*3):
    """
    Check for the completion of a job by searching for its output file and scanning for errors.

    Parameters:
    job_id (str): The job ID to check.
    check_interval (int): Time in seconds between checks.
    timeout (int): Maximum time in seconds to wait for job completion.

    Returns:
    bool: True if job completed successfully, False otherwise.
    """

    if local_output is not None:
        state = check_contents_for_error(local_output)
        if state is None:
            raise Exception('Unexpected output from job')
        else:
            return state

    start_time = time.time()
    output_file = f'Report_islands-{job_id}.out'

    while True:
        # Check if the timeout is reached
        if time.time() - start_time > timeout:
            print("Timeout reached while waiting for job completion.")
            return False

        # Check if the output file exists
        if os.path.exists(output_file):
            with open(output_file, 'r') as file:
                contents = file.read()
                state = check_contents_for_error(contents)
                if state is None:
                    pass
                else:
                    return state

        # Wait for some time before checking again
        time.sleep(check_interval)
        print(f'\t‣ Waiting on check4job_completion LLM job: {job_id} Time: {round(time.time() - start_time)}s', flush=True)


def unpackIslands(num_islands, checkpoints) -> list[islands.Island]:
    islands = []
    for i in range(num_islands):
        curr_llm = ISLAND_LLMS[i]
        print("Unpacking island " + curr_llm, flush=True)
        checkpoint_path = os.path.join(checkpoints, "island_" + curr_llm)
        checkpoint, start_gen = islandIntegration.load_checkpoint(folder_name=args.checkpoints)
        
        if checkpoint:
            GLOBAL_DATA = checkpoint["GLOBAL_DATA"]
            GLOBAL_DATA_HIST = checkpoint["GLOBAL_DATA_HIST"]
            GLOBAL_DATA_ANCESTERY = checkpoint["GLOBAL_DATA_ANCESTERY"]
            population = checkpoint["population"]
            hof = checkpoint["hof"]
        else:
            print("Missing Island ", checkpoint_path)
            exit(0)
        
        individuals = []
        for ind in population:
            individual = islands.Individual(ind[0], ind.fitness.values)
            individuals.append(individual)
        
        island = islands.Island(checkpoint_path, individuals)
        islands.append(island)
    return islands

def packIslands(islands: list[islands.Island]):
    # Define the problem
    creator.create("FitnessMulti", base.Fitness, weights=FITNESS_WEIGHTS)  # Adjust weights as needed
    creator.create("Individual", list, fitness=creator.FitnessMulti, file_id=None)

    # Initialize the toolbox
    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual, creator.Individual)
    toolbox.register("population", create_population)
    toolbox.register("evaluate", evalModel)
    toolbox.register("mate", customCrossover)
    toolbox.register("mutate", customMutation, indpb=0.2)
    toolbox.register("select", true_nsga2)

    for island in islands:
        island.path

        population = toolbox.population()



        curr_llm = ISLAND_LLMS[i]
        print("Generating Island " + curr_llm, flush=True)
        checkpoint_path = os.path.join(checkpoints, "island_" + curr_llm)

        checkpoint, start_gen = islandIntegration.load_checkpoint(folder_name=args.checkpoints)

        if checkpoint:
            GLOBAL_DATA = checkpoint["GLOBAL_DATA"]
            GLOBAL_DATA_HIST = checkpoint["GLOBAL_DATA_HIST"]
            GLOBAL_DATA_ANCESTERY = checkpoint["GLOBAL_DATA_ANCESTERY"]
            population = checkpoint["population"]
            hof = checkpoint["hof"]
        else:
            print("Missing Island ", checkpoint_path)
            exit(0)
        
        individuals = []
        for ind in population:
            individual = islands.Individual(ind[0], ind.fitness.values)
            individuals.append(individual)
        
        island = islands.Island(checkpoint_path, individuals)
        islands.append(island)
    
    return islands


def migrateIslands(topology, num_islands, checkpoints):
    # Load checkpoint data for every island
    # add some individuals to other islands
    # Save them back to checkpoints

    islands = unpackIslands(num_islands, checkpoints)
    new_islands = islands.migrate(topology, islands)
    packIslands(new_islands)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Generation')
    # Add arguments
    parser.add_argument('checkpoints', type=str, help='Save Dir')
    parser.add_argument('--num_islands', type=int, help='Number of Islands', default=2)
    # Parse the arguments
    args = parser.parse_args()
    island_script= "src/island_temp_script.sh"
    checkpoints = args.checkpoints
    num_islands = args.num_islands

    if num_islands >= MAX_ISLANDS:
        print("Number of islands exceeds maximum allowed: " + str(MAX_ISLANDS))
        exit(1)


    # initialize the graph topology
    islands_list = []
    for i in range(num_islands):
        curr_llm = ISLAND_LLMS[i]
        checkpoint_path = os.path.join(checkpoints, "island_" + curr_llm)
        island = islands.Island(checkpoint_path, [])
        islands_list.append(island) 
    topology = islands.generate_graph_topology(islands_list, islands.Topology.FULL)

    # start generation
    for gen in range(num_generations):
        print("Starting generation " + str(gen), flush=True)
        job_ids = []
        for i in range(num_islands):
            curr_llm = ISLAND_LLMS[i]
            print("Generating Island " + curr_llm, flush=True)
            checkpoint_path = os.path.join(checkpoints, "island_" + curr_llm)
            
            job_id = submit_run(island_script, PYTHON_BASH_SCRIPT_TEMPLATE_ISLANDS.format(curr_llm, CONDA_ENV, checkpoint_path, curr_llm, HUGGING_FACE_BOOL))
            job_ids.append(job_id)
        
        done = True
        for i in range(len(job_ids)):
            done = check4job_completion(job_ids[i])
            if not done:
                break
        
        if not done:
            print("Error occured in loop, job not done")
            break
            
        if gen % 5 == 0:
            print("Starting island migration on generation " + str(gen), flush=True)
            migrateIslands(topology, num_islands, checkpoints)
    
    print("Finished evolutionary loop")