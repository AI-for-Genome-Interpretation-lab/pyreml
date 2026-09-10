# %% load packages
from meiosim import Population, Arabidopsis, MISSING
import numpy as np
from tqdm import tqdm
import pickle
from pprint import pprint
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import matplotlib.pyplot as plt

mp.set_start_method("fork", force=True)

# %% parameters
''' params '''
seed = 42

''' Genetic architecture '''
n_snp = 10_000
n_qtl = 10_000

''' Breeding population '''
country = 'SWE'

n_founders = 10
n_per_start = 25
n_cores = 10

print(f"n lines: {(n_founders**2 - n_founders)//2  * n_per_start}")

# %%
''' Simulate genotypes'''
pop = Arabidopsis(n_snp + n_qtl, seed = seed)
country_counts = pop.metadata['country'].value_counts()
pop.metadata['country'] = pop.metadata['country'].apply(
    lambda x: x if country_counts[x] >= 20 else 'other'
)
pprint(pop.metadata['country'].value_counts())

pop.plot("country")
plt.show()

# %% 
subpop = pop.subset({"country": country})
print(len(subpop.genotypes))

missing = subpop.genotypes == MISSING
Z = subpop.genotypes.astype(np.float32) - 1.0
Z[missing] = 0.0
C = (Z @ Z.T) / Z.shape[1]

# remove clones / quasi-clones
dup = (np.triu(C, 1) >= 0.90).any(axis=0)
subpop = subpop.subset(list(np.where(~dup)[0]))
print(len(subpop.genotypes))

# remove heterozygous founders
het = (subpop.genotypes == 1).mean(axis=1)
subpop = subpop.subset(list(np.where(het <= 0.01)[0]))
print(len(subpop.genotypes))

del Z, C, missing

# %% make F1 hybrids
founders = subpop.subset(list(range(n_founders)))

starts = founders.cross(selfing=False, n_cores=n_cores)

# %% derive pure lines
def _self_line(args):
    individual, task_seed = args
    local = starts
    local.rng = np.random.default_rng(task_seed)
    line = local.selfing(individual, 8, n_cores=10)
    line.drop_phases()
    return line

individuals = np.repeat(starts.metadata["individual"].values, n_per_start)
task_seeds = np.random.SeedSequence(seed).spawn(len(individuals))
tasks = list(zip(individuals, task_seeds))

# n parallel lines * 5 chromosomes / lines * 2 haplotypes / chr = 10*n CPUs
with ProcessPoolExecutor(max_workers=n_cores // 10) as executor:
    lines = list(tqdm(
        executor.map(_self_line, tasks),
        total=len(tasks),
        desc="Selfing",
    ))

# %% format output
founders.drop_phases()
starts.drop_phases()

founders.metadata['stage'] = 'founder'
starts.metadata['stage'] = 'start'
for line in lines:
    line.metadata['stage'] = 'line'

breedingpop = Population.merge(founders, starts, *lines)
del lines

breedingpop.metadata['population'] = "breedingpop"
breedingpop.metadata['heterozygosity'] = (
    (breedingpop.genotypes == 1).sum(axis=1) * 100 / len(breedingpop.map)
)
pprint(breedingpop.metadata[['individual', 'stage', 'heterozygosity']].value_counts())

# %% check if perfect clones arised
print(f"n individuals: {len(breedingpop.metadata)}")
unique = np.flatnonzero(~breedingpop.metadata['individual'].duplicated().values)
breedingpop = breedingpop.subset(list(unique))
print(f"n individuals: {len(breedingpop.metadata)}")

# %%
snp, qtl = breedingpop.split(n_snp)
del breedingpop

np.savez_compressed(
    "../data/arabido.npz",
    snp = snp.genotypes.astype(np.int8),
    qtl = qtl.genotypes.astype(np.int8),
    individual = snp.metadata["individual"].to_numpy().astype("U"),
)