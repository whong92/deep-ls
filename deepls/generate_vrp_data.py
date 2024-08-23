import os
import pickle

import numpy as np
from deepls.vrp_gcn_model import VRP_STANDARD_PROBLEM_CONF

def generate_vrp_data(dataset_size, vrp_size):
    capacity = VRP_STANDARD_PROBLEM_CONF[vrp_size]['capacity']
    return list(zip(
        np.full(shape=(dataset_size, 2), fill_value=0.5).tolist(),  # Depot location
        np.random.uniform(size=(dataset_size, vrp_size, 2)).tolist(),  # Node locations
        np.ones((dataset_size, vrp_size), dtype=int).tolist(),  # Demand, uniform integer 1 ... 9
        np.full(dataset_size, capacity).tolist()  # Capacity, same for whole dataset
    ))


def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def save_dataset(dataset, filename):

    filedir = os.path.split(filename)[0]

    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


def load_dataset(filename):

    with open(check_extension(filename), 'rb') as f:
        return pickle.load(f)

if __name__=="__main__":
    size = 50
    N = 10000
    vrp_data = generate_vrp_data(N, size)
    save_dataset(vrp_data, f"/home/ong/personal/deep-ls-tsp/data/vrp-data/size-{size}/vrp_data.pkl")