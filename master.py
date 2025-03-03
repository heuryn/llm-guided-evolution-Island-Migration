import os
import argparse
from deap import base, creator, tools
from deap.tools import HallOfFame
import subprocess
import time
import networkx as nx
from islandIntegration import load_checkpoint, save_checkpoint
from islands import Individual, Island, Topology, migrate, generate_graph_topology
from src.cfg.constants import *
from src.utils.print_utils import box_print

def submit_run(tempFile, text):
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

    if not job_id:
        print("Checking for job completion: job_id is None")
        return None

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


def unpackIslands(num_islands, checkpoints) -> list[Island]:
    islands = []
    for i in range(num_islands):
        curr_llm = ISLAND_LLMS[i]
        print("Unpacking island " + curr_llm, flush=True)
        checkpoint_path = os.path.join(checkpoints, "island_" + curr_llm)
        checkpoint, start_gen = load_checkpoint(folder_name=checkpoint_path)
        
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
            individual = Individual(ind[0], ind.fitness.values)
            individuals.append(individual)
        
        island = Island(checkpoint_path, individuals)
        islands.append(island)
    return islands

def packIslands(islands: list[Island], gen: int):
    creator.create("Individual", list, fitness=creator.FitnessMulti, file_id=None)
    for island in islands:
        island_path = island.path
        print("Packing island path" + island_path, flush=True)
        population = []
        for individual in island.individuals:
            ind = creator.Individual([individual.name])
            ind.fitness.values = individual.rank
            population.append(ind)
        
        hof = tools.HallOfFame(hof_size)
        checkpoint_data = {
            "GLOBAL_DATA": {},
            "GLOBAL_DATA_HIST": {},
            "population": population,
            "hof": hof,
            "GLOBAL_DATA_ANCESTERY": {},
        }
        save_checkpoint(gen, island_path, checkpoint_data)

        

import collections
def print_swaps(before1, before2, after1, after2):
    """
    Given two arrays before the swap (before1, before2) and two arrays after the swap (after1, after2),
    prints which entries were swapped from Array 1 to Array 2 and vice versa,
    and displays the total count of swapped entries.
    
    The function compares the counts of each element in the "before" arrays with the "after" arrays.
    """
    # Count occurrences in each list
    counter1_before = collections.Counter(before1)
    counter1_after = collections.Counter(after1)
    counter2_before = collections.Counter(before2)
    counter2_after = collections.Counter(after2)

    # Determine items that left Array 1 (i.e. swapped from Array 1 to Array 2)
    swapped_1_to_2 = {}
    for item, count in counter1_before.items():
        # If the item appears less in after1, it means some copies moved out.
        diff = count - counter1_after.get(item, 0)
        if diff > 0:
            swapped_1_to_2[item] = diff

    # Determine items that left Array 2 (i.e. swapped from Array 2 to Array 1)
    swapped_2_to_1 = {}
    for item, count in counter2_before.items():
        diff = count - counter2_after.get(item, 0)
        if diff > 0:
            swapped_2_to_1[item] = diff

    # Calculate total number of swapped entries.
    total_swapped = sum(swapped_1_to_2.values()) + sum(swapped_2_to_1.values())

    # Print out the results
    print("Entries swapped from Array 1 to Array 2:")
    if swapped_1_to_2:
        for item, count in swapped_1_to_2.items():
            print(f"  {item}: {count}")
    else:
        print("  None")

    print("\nEntries swapped from Array 2 to Array 1:")
    if swapped_2_to_1:
        for item, count in swapped_2_to_1.items():
            print(f"  {item}: {count}")
    else:
        print("  None")

    print(f"\nTotal number of swapped entries: {total_swapped}")



def migrateIslands(topology, num_islands, checkpoints, gen):
    # Load checkpoint data for every island
    # add some individuals to other islands
    # Save them back to checkpoints

    # array of class Island 
    print("UNPACKING ISLANDS")
    islands = unpackIslands(num_islands, checkpoints)

    print()
    array1_before = []
    island1 = islands[0]
    for individual in island1.individuals:
        array1_before.append(individual.name)
    array2_before = []
    island2 = islands[1]
    for individual in island2.individuals:
        array2_before.append(individual.name)
    
    print("MIGRATING INDIVIDUALS")
    migrate(topology, islands)

    array1_after = []
    island1 = islands[0]
    for individual in island1.individuals:
        array1_after.append(individual.name)
    array2_after = []
    island2 = islands[1]
    for individual in island2.individuals:
        array2_after.append(individual.name)
    
    print(" ----------- print_swaps output ----------- ")
    print()
    print_swaps(array1_before, array2_before, array1_after, array2_after)

    print("PACKING ISLANDS")
    print()
    packIslands(islands, gen)



def submit_mutate_prompts(llm_model, n=5):
    prompt_job_ids = []
    templates = np.random.choice(glob.glob(f'{ROOT_DIR}/templates/FixedPrompts/*/*.txt'), n)
    file_path = './mutate_prompts_temp.sh'
    for i, template in enumerate(templates):
        python_runline = f"python src/llm_prompt_mutation.py --llm_model {llm_model} --template {template}"
        script = LLM_BASH_SCRIPT_TEMPLATE.format(LLM_GPU, python_runline)

        with open(file_path, 'w') as file:
            file.write(script)
        print(f"\t‣ Bash Script Saved to {file_path}")

        job_id = None
        successful_sub_flag = False
        result = subprocess.run([RUN_COMMAND, file_path], capture_output=True, text=True)
        if result.returncode == 0:
            print("\t‣ Script Submitted Successfully.\n\t‣ Output:", result.stdout.strip())
            successful_sub_flag = True
            job_id = result.stdout.split('job ')[-1].strip()
        else:
            print("\t‣ Failed to Submit script.\n\t‣ Error:", result.stderr.strip())
            successful_sub_flag = False
            job_id = None
        
        prompt_job_ids.append(job_id)
    return prompt_job_ids
    



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
        island = Island(checkpoint_path, [])
        islands_list.append(island) 
    topology = generate_graph_topology(islands_list, Topology.FULL)


    '''
        jack's migration test
    

    box_print("Start of Test")
    print("checkpoints: ", checkpoints)
    print("num islands: ", num_islands)
    print("topology: ", topology)

    print("Starting island migration", flush=True)
    migrateIslands(topology, num_islands, checkpoints, 0)


    box_print("End of Test")
    exit(0)

    '''

    # start generation
    for gen in range(num_generations):
        print("Starting generation " + str(gen), flush=True)
        job_ids = []

        # submit island generation jobs
        for i in range(num_islands):
            curr_llm = ISLAND_LLMS[i]
            print("Generating Island " + curr_llm, flush=True)
            checkpoint_path = os.path.join(checkpoints, "island_" + curr_llm)
            
            job_id = submit_run(island_script, PYTHON_BASH_SCRIPT_TEMPLATE_ISLANDS.format(curr_llm, CONDA_ENV, checkpoint_path, curr_llm))
            job_ids.append(job_id)
        
        # check island generation jobs for completion
        done = True
        for i in range(len(job_ids)):
            done = check4job_completion(job_ids[i])
            if not done:
                break
        
        if not done:
            print("Error occured in loop, job not done")
            break

        '''
        # mutate prompts
        print("Mutating Prompts")
        prompt_job_ids = submit_mutate_prompts(LLM_MIXTRAL)
        done = True
        for i in range(len(prompt_job_ids)):
            done = check4job_completion(prompt_job_ids[i])
            if not done:
                break

        if not done:
            print("Error occured in loop, job not done")
            break
        '''

        # migrate individuals between islands
        if gen % 1 == 0:
            print("Starting island migration on generation " + str(gen), flush=True)
            migrateIslands(topology, num_islands, checkpoints, gen)
    
    print("Finished evolutionary loop")