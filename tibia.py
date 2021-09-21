#
# Converts nodes --> coordinates and analyzes the
# limits of the footprints relative to the mean 
# bone in a percentage like the Wolf paper
# - TIBIA only
#
# M. Sale
# 17 June 2021
#

import numpy as np
import os
import open3d as o3d
import pandas as pd
import csv

dimensions = ['x','y','z']

def load_ply(path):
    # Load the mean bone's points (point cloud) from a file
    # open 3d ply --> pcd
    print("getting bone point cloud...")
    try:
        bone_ply = o3d.io.read_point_cloud(path)
        bone_points = np.asarray(bone_ply.points)
        return bone_points
        
    except:
        print("ERROR getting the bone point cloud")
        return -1


def load_footprint(path):
    """
    Loads footprint coordinates as a pandas dataframe
    Returns each x,y,z coordinates as integers
    """
    data = pd.read_csv(path, delim_whitespace=True, names=dimensions)
    x = data['x'].astype('int')
    y = data['y'].astype('int')
    z = data['z'].astype('int')
    return x,y,z


def load_centroid(path):
    """
    Load in a text file of the centroid (xyz coords)
    """
    data = np.loadtxt(path)
    return data


def load_csv_nodes(path):
    """
    Given .csv of nodes and their frequencies (i.e. heatmap)
    Loads the path and returns just a list of the node numbers involved
    """
    df = pd.read_csv(path, names=["Node", "Frequency"])

    df["Node"] = df["Node"].astype('int')
    frequencies = df["Frequency"]

    node_list = df["Node"].tolist()

    return node_list



def get_footprint_bounds(bone_points, footprint_coords):
    """
    @Input: file ID: string
            bone_points as a numpy array
            footprint coords [x,y,z] as ints (from load_footprint)
    @Output: Array of the four bounds = [minAP, maxAP, minML, maxML]
    """
    print('-------------------')
    print(footprint_coords)
    footprintX = footprint_coords[0]
    footprintY = footprint_coords[1]
    footprintZ = footprint_coords[2]

    # Get the max and min coordinates for the AP and ML axes
    tibia = pd.DataFrame(data=bone_points, columns=dimensions)
    # x bounds (ANTERIOR - POSTERIOR)
    minX = tibia["x"].min()
    maxX = tibia["x"].max()
    rangeX = maxX-minX # this is the 100% measure

    # y bounds (DISTAL)
    minY = tibia["y"].min()
    maxY = tibia["y"].max()

    # z bounds (MEDIAL - LATERAL)
    minZ = tibia["z"].min()
    maxZ = tibia["z"].max()
    rangeZ = maxZ-minZ # this is the 100% measure


    # - FOOTPRINTS

    # * X: Anterior - Posterior 
    # Calculate min and max footprint bounds as a percentage of bone bounds
    footprint_minX = footprintX.min()
    footprint_maxX = footprintX.max()

    # Calculate the limits as a percentage of the total bounds
    minAP = (footprint_minX-minX)/rangeX
    maxAP = (footprint_maxX-minX)/rangeX

    # * Z Medial - Lateral
    # Calculate min and max footprint bounds as a percentage of bone bounds
    footprint_minZ = footprintZ.min()
    footprint_maxZ = footprintZ.max()

    # Calculate the limits as a percentage of the total bounds
    minML = (footprint_minZ-minZ)/rangeZ
    maxML = (footprint_maxZ-minZ)/rangeZ

    print('---------------')

    # Save the four values
    return [minAP, maxAP, minML, maxML]


def save(path, data):
    # Write the saved values to a csv
    try:
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerows(data.items()) # array 
            # w.writerows(data)  # list of lists
            print('File write successful')
    except:
        print('ERROR writing data to file.')


def save_centroids(path, data):
    """
    NEW function Sept 2021
    saves array of centroid data to csv
    """
    # Write the saved values to a csv
    try:
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerows(data.items()) # array 
            print('File write successful')
    except:
        print('ERROR writing data to file.')


def save_footprint(path, data):
    """
    NEW function Sept 2021
    Identical to save(). This function is available for better code clarity and use. 
    saves footprint bounds data from list of lists to csv
    """
    # Write the saved values to a csv
    try:
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerows(data)  # list of lists
            print('File write successful')
    except:
        print('ERROR writing data to file.')
