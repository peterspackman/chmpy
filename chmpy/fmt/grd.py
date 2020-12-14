import logging
import numpy as np


def parse_grd_file(filename):
    contents = {}

    with open(filename) as f:
        contents["header"] = [f.readline().strip(), f.readline().strip()]
        f.readline()
        f.readline()
        contents["npts"] = tuple(int(x) for x in f.readline().split())
        contents["origin"] = tuple(float(x) for x in f.readline().split())
        contents["dimensions"] = tuple(float(x) for x in f.readline().split())
        f.readline()
        nobjects = int(f.readline())
        objects = []
        for i in range(nobjects):
            objects.append(f.readline())
        f.readline()
        nconnections = int(f.readline())
        connections = []
        for i in range(nconnections):
            connections.append(f.readline())
        f.readline()
        contents["data"] = np.loadtxt(f)
    return contents
