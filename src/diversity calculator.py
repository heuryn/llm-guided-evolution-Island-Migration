import math
import numpy as np
from collections import defaultdict, Counter

####################################
# Self-CIDEr Helper Functions
####################################

def extract_ngrams(text, n):
    """
    Extracts all n-grams (as tuples) from the text using whitespace tokenization.
    """
    #Splits text into tokens (whitespace-based)
    tokens = text.split()
    ngrams = []
    if len(tokens) < n: #Handles edge case for short texts
        return ngrams
    #Generates overlapping n-grams as tuples
    for i in range(len(tokens) - n + 1): 
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)
    return ngrams

def compute_tf(text, n):
    """
    Computes term frequency for n-grams of order n in the text.
    Normalizes the counts by the total number of n-grams.
    """
    ngrams = extract_ngrams(text, n)
    # Uses Counter for efficient frequency counting
    tf = Counter(ngrams)
    total = sum(tf.values())
    if total > 0:
        for key in tf:

            tf[key] /= total
    return tf

def compute_idf(population, n, epsilon=1e-6):
    """
    Computes inverse document frequency for n-grams of order n in a population.
    Uses the formula: idf = log((N + 1) / (df + 1) + epsilon)
    """
    N = len(population)
    df = defaultdict(int)
    for text in population:

        ngrams = set(extract_ngrams(text, n))
        for ngram in ngrams:
            df[ngram] += 1
    idf = {}
    for ngram, count in df.items():
        idf[ngram] = math.log((N + 1) / (count + 1) + epsilon)
    return idf

def compute_combined_idf(population, ngram_range=(1,4)):
    """
    Computes a combined IDF dictionary for each n in ngram_range.
    Returns a dict mapping (n, ngram) to its IDF weight.
    """
    combined_idf = {}
    for n in range(ngram_range[0], ngram_range[1]+1):
        idf_n = compute_idf(population, n)
        for ngram, weight in idf_n.items():
            combined_idf[(n, ngram)] = weight

    return combined_idf

def compute_tf_idf(text, idf, n):
    """
    Computes the TF-IDF vector for a given text and n-gram order n.
    """
    tf = compute_tf(text, n)
    tf_idf = {}

    for ngram, freq in tf.items():
        tf_idf[ngram] = freq * idf.get(ngram, 0.0)
    return tf_idf

def cosine_similarity(vec1, vec2):
    """
    Computes the cosine similarity between two TF-IDF vectors.
    """
    dot = sum(vec1.get(key, 0.0) * vec2.get(key, 0.0) for key in vec1)
    norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
    
    norm2 = math.sqrt(sum(v**2 for v in vec2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def compute_self_cider_similarity(code1, code2, combined_idf, ngram_range=(1,4)):
    """
    Computes a Self-CIDErinspired similarity between two code strings.
    For each n in ngram_range, computes cosine similarity using the corresponding TF-IDF vectors,
    then returns the average similarity.
    """
    similarities = []
    for n in range(ngram_range[0], ngram_range[1]+1):
        # Filter IDF weights for the current n-gram order
        idf_n = {ngram: weight for (order, ngram), weight in combined_idf.items() if order == n}
        
        vec1 = compute_tf_idf(code1, idf_n, n)
        vec2 = compute_tf_idf(code2, idf_n, n)
        sim = cosine_similarity(vec1, vec2)
        similarities.append(sim)
    return sum(similarities) / len(similarities) if similarities else 0.0

def compute_self_cider_distance(code1, code2, combined_idf, ngram_range=(1,4)):
    """
    Converts Self-CIDEr similarity into a distance metric.
    Distance is defined as 1 - similarity.
    """
    similarity = compute_self_cider_similarity(code1, code2, combined_idf, ngram_range)
    return 1 - similarity

####################################
# Diversity Quantifier Functions
####################################

def computeWithinIslandDiversity(population, ngram_range=(1,4)):
    """
    Calculates the average pairwise Self-CIDEr distance (diversity) within an island.
    """
    n = len(population)
    if n < 2:
        return 0.0
    combined_idf = compute_combined_idf(population, ngram_range)
    total_distance = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            distance = compute_self_cider_distance(population[i], population[j], combined_idf, ngram_range)
            total_distance += distance
            count += 1
    return total_distance / count

def compute_diversity_contribution(individual, population, ngram_range=(1,4)):
    """
    Calculates how unique an individual is relative to others in its island.
    Returns the average Self-CIDEr distance from this individual to all others.
    """
    if len(population) < 2:
        return 0.0
    combined_idf = compute_combined_idf(population, ngram_range)
    distances = [
        compute_self_cider_distance(individual, other, combined_idf, ngram_range)
        for other in population if other != individual
    ]
    return sum(distances) / len(distances) if distances else 0.0

def computeAmongIslandsDiversity(island1, island2, ngram_range=(1,4)):
    """
    Calculates the average Self-CIDEr distance between individuals from two different islands.
    """
    combined_population = island1 + island2
    if not combined_population:
        return 0.0
    combined_idf = compute_combined_idf(combined_population, ngram_range)
    distances = []
    for code1 in island1:
        for code2 in island2:
            distances.append(compute_self_cider_distance(code1, code2, combined_idf, ngram_range))
    return sum(distances) / len(distances) if distances else 0.0

def migration_decision(population, fitness_scores, alpha=0.5, beta=0.5, migration_rate=0.2, ngram_range=(1,4)):
    """
    Determines which individuals should migrate based on a composite score.
    
    The migration score for each individual is:
         migration_score = alpha * fitness + beta * diversity_contribution
    where diversity_contribution is the average Self-CIDEr distance to other individuals.
    
    migration_rate is the fraction of the population selected for migration.
    """
    migration_scores = []
    for individual, fitness in zip(population, fitness_scores):

        diversity_contrib = compute_diversity_contribution(individual, population, ngram_range)
        migration_score = alpha * fitness + beta * diversity_contrib
        migration_scores.append(migration_score)
    
    num_to_migrate = max(1, int(len(population) * migration_rate))
    indices = np.argsort(migration_scores)[-num_to_migrate:]
    return [population[i] for i in indices]

# Example


if __name__ == "__main__":
    # Example island: individuals represented as code strings.
    island_population = [
        "def foo():\n    return 1",
        "def foo():\n    return 2",
        "def foo():\n    return 3",
        "def foo():\n    return 4"
    ]
    # Hypothetical fitness scores (e.g., from model evaluation)
    fitness_scores = [0.9, 0.8, 0.85, 0.95]
    
    # 1. Compute diversity within an island.
    intra_div = computeWithinIslandDiversity(island_population)
    print("Intra-Island Diversity (Self-CIDEr):", intra_div)
    
    # 2. Decide which individuals should migrate.
    migrants = migration_decision(island_population, fitness_scores, alpha=0.7, beta=0.3, migration_rate=0.5)
    print("\nMigration Candidates (Self-CIDEr):")
    for candidate in migrants:
        print(candidate)
    
    # 3. Compute diversity between two islands.
    island_population2 = [
        "def bar():\n    return 'a'",
        "def bar():\n    return 'b'",
        "def bar():\n    return 'c'"
    ]
    inter_div = computeAmongIslandsDiversity(island_population, island_population2)
    print("\nInter-Island Diversity (Self-CIDEr):", inter_div)
