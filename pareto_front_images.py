import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import math

output_dir = 'data/pareto_fronts'
os.makedirs(output_dir, exist_ok=True)

# Helper function to extract fitness values from a given dataset
def extract_fitness_values(dataset):
    fitness_values = []
    for attributes in dataset.values():
        if 'fitness' in attributes:
            # Remove individuals with fitness (-inf, inf)
            if attributes['fitness'] != (-float('inf'), float('inf')):
                fitness_values.append(attributes['fitness'])
    return fitness_values

# Function to load data and extract fitness values for a given generation
def get_fitness_values_for_generation(gen_number):
    file_path = f'data/global_data/global_gen_{gen_number}.pkl'
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    # Extract fitness values from GLOBAL_DATA and GLOBAL_DATA_HIST
    global_data_fitness = extract_fitness_values(data.get('GLOBAL_DATA', {}))
    global_data_hist_fitness = extract_fitness_values(data.get('GLOBAL_DATA_HIST', {}))

    return global_data_fitness, global_data_hist_fitness

# Function to identify Pareto frontier
def pareto_frontier(rates):
    # Sort by first objective in descending order and second objective in ascending order
    sorted_rates = sorted(rates, key=lambda x: (-x[0], x[1]))
    
    pareto_front = [sorted_rates[0]]
    for rate in sorted_rates[1:]:
        if rate[1] < pareto_front[-1][1]:
            pareto_front.append(rate)
    return pareto_front

# Function to create and save Pareto front plot for a generation
def plot_pareto_front(global_data_fitness, global_data_hist_fitness, gen_number):
    output_path = os.path.join(output_dir, f'pareto_gen_{gen_number}.png')
    if os.path.exists(output_path):
        print(f"Pareto front image for generation {gen_number} already exists.")
        return
    # Combine fitness values from both datasets
    global_data_fitness = [t for t in (global_data_fitness or []) if isinstance(t, (tuple, list)) and len(t) == 2 and all(v is not None for v in t)]
    global_data_hist_fitness = [t for t in (global_data_hist_fitness or []) if isinstance(t, (tuple, list)) and len(t) == 2 and all(v is not None for v in t) ]
    
    all_fitness_values = global_data_fitness + global_data_hist_fitness

    '''
    bins = 10
    xs, ys = zip(*all_fitness_values)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    x_edges = np.linspace(xmin, xmax, bins+1)
    y_edges = np.linspace(ymin, ymax, bins+1)
    '''
    
    if not all_fitness_values:
        print(f"No valid fitness values for generation {gen_number}")
        return
    if not global_data_fitness and not global_data_hist_fitness:
        print(f"No valid points to plot for gen {gen_number}")
        return
    
    # Extract x and y coordinates for global data fitness
    global_xs, global_ys = zip(*global_data_fitness) if global_data_fitness else ([], [])
    
    # Extract x and y coordinates for global data hist fitness
    hist_xs, hist_ys = zip(*global_data_hist_fitness) if global_data_hist_fitness else ([], [])
    
    # Determine the Pareto front
    pareto_front = pareto_frontier(all_fitness_values)
    pareto_xs, pareto_ys = zip(*pareto_front) if pareto_front else ([], [])
    print(f"Pareto front for generation {gen_number}: {pareto_front}")
    # Get points for the Pareto frontier line
    pareto_xs = list(pareto_xs)
    pareto_ys = list(pareto_ys)

    # Create the plot with a stretched x-axis through the figsize parameter
    plt.figure(figsize=(10, 5))  # Width is 10 and height is 5
    plt.scatter(global_xs, global_ys, s=10, alpha=0.5, label='Global Data Points', color='blue')
    plt.scatter(hist_xs, hist_ys, s=10, alpha=0.5, label='Hist Data Points', color='green')
    plt.scatter(pareto_xs, pareto_ys, s=15, color='red', label='Pareto Front Points')

    # Add a star marker at the specified coordinates
    special_x = 0.9252
    special_y = 518230.0
    plt.scatter([special_x], [special_y], color='gold', s=100, marker='*', label='ExquisiteNetV2')

    if pareto_xs and pareto_ys:
        pareto_xs = [1] + pareto_xs + [0]
        pareto_ys = [1e7] + pareto_ys + [0]
    
    '''
    for xv in x_edges:
        plt.axvline(x=xv, linewidth=0.6, alpha=0.45)
    for yv in y_edges:
        plt.axhline(y=yv, linewidth=0.6, alpha=0.45)
    '''

    plt.step(pareto_xs, pareto_ys, color='gray', where='post', label='Pareto Front Line')
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.title(f'Pareto Front - Generation {gen_number}')
    plt.legend()
    
    # Set the y-axis to a logarithmic scale
    plt.yscale('log')
    
    # Set x and y axis limits
    plt.xlim(0.8, 1)
    #plt.xlim(xmin, 1)
    plt.ylim(1e4, 1e7)
    #plt.yscale('log')
    #plt.ylim(max(ymin, 1e-12), ymax)
    
    # Save the plot
    output_path = os.path.join(output_dir, f'pareto_gen_{gen_number}.png')
    plt.savefig(output_path)
    plt.close()

##for new, normalized pareto front
def normalized_values(vals, x_min, x_max, p_log_min, p_log_max):
    out = []
    dx = max(1e-12, x_max - x_min)
    dy = max(1e-12, p_log_max - p_log_min)
    for x, y in vals:
        xn = (x-x_min)/dx
        ylog = math.log10(max(y, 1e-12))
        yn = (ylog - p_log_min)/dy

        xn = min(max(xn, 0.0), 1.0 - 1e-12)
        yn = min(max(yn, 0.0), 1.0 - 1e-12)
        out.append((xn, yn))

    return out

def plot_normalized_grid(global_data_fitness, global_data_hist_fitness, gen_number,
                         bins=10,
                         x_min=0.80, x_max=1.00,
                         p_log_min=4.0, p_log_max=8.0,
                         output_dir='data/pareto_fronts'):
    """Plots points normalized to [0,1]^2 with a fixed bins×bins grid."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    global_data_fitness = [
        tuple(t) for t in (global_data_fitness or [])
        if isinstance(t, (tuple, list)) and len(t) == 2 and all(v is not None for v in t)
    ]
    global_data_hist_fitness = [
        tuple(t) for t in (global_data_hist_fitness or [])
        if isinstance(t, (tuple, list)) and len(t) == 2 and all(v is not None for v in t)
    ]
    
    all_vals = [tuple(t) for t in (global_data_fitness or []) if len(t)==2] + \
               [tuple(t) for t in (global_data_hist_fitness or []) if len(t)==2]
    if not all_vals:
        return

    # Use performance-friendly fixed bounds: acc in [x_min,x_max], params in [10^p_log_min, 10^p_log_max]
    #y_min, y_max = 10**p_log_min, 10**p_log_max

    ys = [y for _, y in all_vals]
    print(f"[Gen {gen_number}] Params range: min={min(ys):.2e}, max={max(ys):.2e}")
    Gn = normalized_values(global_data_fitness or [], x_min, x_max, p_log_min, p_log_max)
    Hn = normalized_values(global_data_hist_fitness or [], x_min, x_max, p_log_min, p_log_max)
    An = Gn + Hn

    # Pareto on normalized coords (same dominance as original for linear scaling on x and monotone y)
    pf = pareto_frontier(An)
    pf_x, pf_y = zip(*pf) if pf else ([], [])

    # Grid edges in normalized space
    edges = np.linspace(0.0, 1.0, bins + 1)

    plt.figure(figsize=(6.5, 6))
    if Gn:
        gx, gy = zip(*Gn)
        plt.scatter(gx, gy, s=12, alpha=0.6, label='GLOBAL_DATA')
    if Hn:
        hx, hy = zip(*Hn)
        plt.scatter(hx, hy, s=12, alpha=0.6, label='GLOBAL_DATA_HIST')
    if pf_x:
        plt.scatter(pf_x, pf_y, s=20, edgecolors='k', linewidths=0.5, label='Pareto (norm)')

        sx = [1.0] + list(pf_x) + [0.0]
        sy = [1.0] + list(pf_y) + [0.0]
        plt.step(sx, sy, where='post', linewidth=1.0, label='Pareto Line (norm)')

    # Draw 10×10 grid in normalized space
    for v in edges:
        plt.axvline(v, linewidth=0.6, alpha=0.45)
        plt.axhline(v, linewidth=0.6, alpha=0.45)

    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xticks(edges); plt.yticks(edges)
    plt.grid(False)
    plt.xlabel('Normalized accuracy')
    plt.ylabel('Normalized params (fixed log-range)')
    plt.title(f'Normalized Pareto Front Grid — Gen {gen_number}')
    plt.legend(fontsize=8, loc='best')
    out = os.path.join(output_dir, f'pareto_norm_grid_gen_{gen_number}.png')
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


# Create and save Pareto front images for each generation from 1 to 27
for gen_number in range(1, 1000):
    global_data_fitness, global_data_hist_fitness = get_fitness_values_for_generation(gen_number)
    if global_data_fitness or global_data_hist_fitness:  # Only plot if there are fitness values to plot
        plot_pareto_front(global_data_fitness, global_data_hist_fitness, gen_number)
        plot_normalized_grid(global_data_fitness, global_data_hist_fitness, gen_number)
    print(f'Pareto front for generation {gen_number} processed.')

print('Pareto front images for all generations have been created and saved.')
