import random
import pandas
import numpy
import sklearn.cluster

def read_sampled_file(data_file, N, random_sampling=False):
    """
    
    """
    num_lines = sum(1 for l in open(data_file))
    sample_size = int(num_lines / N)

    if random_sampling:
        skip_index = random.sample(range(1, num_lines), num_lines - sample_size)  # random sample
    else:
        skip_index = [x for x in range(1, num_lines) if x % N != 0] # every N lines
    
    df = pandas.read_csv(data_file, sep='|', skiprows=skip_index)    
    return df


def add_jitter(values):
    """
    
    """
    return values + numpy.random.normal(1, 0.4, len(values))

def elbow_method(X, num_clusters=10):
    """
    
    """
    d = []
    for n in range(1, num_clusters):
        kmeans = sklearn.cluster.KMeans(n_clusters=n).fit(X)
        d.append(kmeans.inertia_)
    return d

def kmeans_cluster(X, num_clusters):
    """
    
    """
    kmeans = sklearn.cluster.KMeans(n_clusters=num_clusters).fit(X)
    labels = kmeans.predict(X)
    C = kmeans.cluster_centers_
    return (labels, C)