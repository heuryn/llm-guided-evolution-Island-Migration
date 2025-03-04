import collections
import math
import time

"""
Printing Functions: Move to Other File
"""
def print_population(offspring, global_data):
    box_print(" Poplutation Info:", print_bbox_len=60, new_line_end=False)
    print(f'🧬 Poplutation Size 🧬: {len(offspring)}')
    for ind in offspring:
        gene_id = ind[0]
        print(f'Gene: {gene_id}')
        if gene_id in global_data:
            print_job_info(global_data[gene_id], short=True)

def print_scores(population, fitness_weights):
    num_objectives = len(fitness_weights)
    objective_scores = [[] for _ in range(num_objectives)]

    # Collect scores for each objective
    for ind in population:
        for i, f in enumerate(ind.fitness.values):
            objective_scores[i].append(f)

    # Calculate and print stats for each objective
    box_print("SCORES", print_bbox_len=110, new_line_end=True)
    for i in range(num_objectives):
        fits = objective_scores[i]
        fits = [x for x in objective_scores[i] if math.isfinite(x)]
        length = len(fits)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        direction = "Maximize" if fitness_weights[i] > 0 else "Minimize"
        
        print(f"Objective {i+1} ({direction}):")
        print(f"  Min: {min(fits)}")
        print(f"  Max: {max(fits)}")
        print(f"  Avg: {mean}")
        print(f"  Std: {std}")
        print()

def box_print(txt, print_bbox_len=110, new_line_end=True):
    # just for logging 
    def replace_middle(v, x):
        start_pos = (len(v) - len(x)) // 2
        return v[:start_pos] + x + v[start_pos + len(x):]
    
    v = "*" + " " * (print_bbox_len - 2) + "*"
    end = '\n' if new_line_end else ''
    print_result = "\n" + "*" * print_bbox_len + "\n" + replace_middle(v, txt) + "\n" + "*" * print_bbox_len + end
    print(print_result, flush=True)
    
def print_job_info(job_dict, short=False):
    print(f"\t‣ Fitness: {job_dict['fitness']}, Submission Flag: {job_dict['sub_flag']}")
    print(f"\t‣ Runtime: {round((time.time()-job_dict['start_time'])/60)} min, Status: {job_dict['status']}")
    if short is False:
        try:
            print(f"\t‣ LLM Job-ID: {job_dict['job_id']}, Model Job-ID: {job_dict['results_job']}")
        except:
            print(f"\t‣ {job_dict}")
    else:
        print(f"\t‣ LLM Job-ID: {job_dict['job_id']}")

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