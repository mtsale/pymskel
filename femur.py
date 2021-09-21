#
# * Use: from pymskel import femur2
#
#
# M. Sale July 8 2021
#

import pandas as pd
import numpy as np
import os,sys
import gias2
from scipy.spatial import cKDTree
import mayavi.mlab as mlab
from gias2.visualisation import fieldvi_custom as fieldvi
from gias2.mesh import vtktools
import open3d as o3d
import csv
from operator import itemgetter
import matplotlib.pyplot as plt
import math
from scipy.interpolate import splev, splrep, sproot, spalde, BSpline


def load_ply(file):
    """
    Loads a .ply file path of a bone located at a specified path. 
    Uses the gias2 vtk loader. Could use open3d instead if 
    gias decides to break.

    Returns the xyz point cloud as a numpy array
    """
    # femur = vtktools.loadpoly(f'../Data/Femur/02_Femur_PCA/0244_HD_SSM_fitted_rbfreg_rigidreg.ply')
    poly = vtktools.loadpoly(file)
    points = poly.v
    return points


def load_footprint_txt(path):
    """
    Loads a txt file of footprint nodes
    Returns nodes as integers
    """
    footprint_nodes = np.loadtxt(path).astype('int')
    return footprint_nodes


def load_footprint_csv(path):
    """
    Loads footprint coordinates as a pandas dataframe
    Returns each x,y,z coordinates as integers
    """
    dimensions = ["x", "y", "z"]
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

def convert_xyz_to_node(bone_points, xyz_point):
    """
    Get node number of coordiante on a bone
    """
    b = bone_points
    a = xyz_point

    xtree = cKDTree(b)
    ad, ai = xtree.query(a, k=1)

    # temp = []
    # for i in range(1):
    #     xyz_bone = b[ai[i]]
    #     temp.append(xyz_bone)

    # bone_xyz_points, bone_xyz_mpts =  np.array(temp), ai

    # # Remove duplicates
    # mpts_unique = np.array(list(dict.fromkeys(bone_xyz_mpts)))
    print(ai)
    return ai


def get_pcd(points):
    """
    Converts array of points into open3D point cloud
    """
    # Load point cloud from points
    bone_pcd = o3d.geometry.PointCloud()
    bone_pcd.points = o3d.utility.Vector3dVector(points)
    return bone_pcd


def rotate_to_lateral(bone_points):
    """
    Transform the knee to be lateral reference view
    @Input: Bone points 3D numpy array
    @Output: Bone points 3D in the new transformed subspace
    """

    # Can set a new matrix here if you need a different (or better) view
    # Meshlab --> save camera view ---> open file and get rotation matrix
    ROTATION_MATRIX =   [[0.630644, -0.757998, 0.166513, 0.00000000],
                        [-0.756577, -0.648273, -0.0856336, 0.00000000],
                        [0.172856, -0.0719752, -0.982314, 0.00000000],
                        [0.00000000, 0.00000000, 0.00000000, 1.00000000]]

    # Make into homogenous coordinates with a w dimension
    bone_points = np.vstack([bone_points.T, np.ones(bone_points.shape[0])])
    transformed_bone = np.dot(bone_points.T, ROTATION_MATRIX)

    # Drop 4th dimension from affine result
    transformed_bone = transformed_bone[:, [0,1,2]]
    
    return transformed_bone

def rotate_footprint(bone_points):
    """
    UNUSED
    Transform the knee to be lateral reference view
    @Input: Bone points 3D numpy array
    @Output: Bone points 3D in the new transformed subspace
    """

    # Can set a new matrix here if you need a different (or better) view
    # Meshlab --> save camera view ---> open file and get rotation matrix
    ROTATION_MATRIX =   [[0.630644, -0.757998, 0.166513, 0.00000000],
                        [-0.756577, -0.648273, -0.0856336, 0.00000000],
                        [0.172856, -0.0719752, -0.982314, 0.00000000],
                        [0.00000000, 0.00000000, 0.00000000, 1.00000000]]

    # Make into homogenous coordinates with a w dimension
    transformed_bone = np.dot(ROTATION_MATRIX, bone_points)
    
    return transformed_bone.T


def get_bounding_box_xyz(bone_points):
    """
    Returns x,y,z values as lists of the axis aligned 
    bounding box around the input bone.
    """
    # Load point cloud from points
    bone_pcd = o3d.geometry.PointCloud()
    bone_pcd.points = o3d.utility.Vector3dVector(bone_points)

    # apply axis aligned bounding box
    aa_boundingBox = bone_pcd.get_axis_aligned_bounding_box()
    boundBox = aa_boundingBox.get_extent() 
    # print("Pre-Crop BBox Extent: " , boundBox)

    # 8 points which define the bounding box
    boundBox_8points = aa_boundingBox.get_box_points()

    # DEBUG print statements
    # print(np.asarray(boundBox_8points))
    # o3d.visualization.draw_geometries([bone_pcd, aa_boundingBox])

    x_vals = []
    y_vals = []
    z_vals = []

    for i in boundBox_8points:
        x_vals.append(i[0])
        y_vals.append(i[1])
        z_vals.append(i[2])

    return x_vals, y_vals, z_vals


def get_min_max(value_list):
    """
    Return min and max of values in the list
    """
    _min = min(value_list)
    _max = max(value_list)

    return _min, _max


def get_notch_apex(bone_points):
    """
    Numerically finds the apex of the intercondylar notch
    Returns NOTCH = [x,y,z]

    Algo: Find largest y_minima in mid 40% of z extent
    """
    bone_pcd = get_pcd(bone_points)
    # print('\nGetting notch apex...\n')
    # o3d.visualization.draw_geometries([bone_pcd])

    # Get the axis bounds
    x_vals, y_vals, z_vals = get_bounding_box_xyz(bone_points)

    z_min, z_max = get_min_max(z_vals)
    z_extent = z_max-z_min

    # We only want to screen the middle 40% of the bone (region where the notch is)
    # Z crop limits
    z_low = z_min + z_extent*0.30
    z_high = z_max - z_extent*0.30

    z_cropBox = []
    # iterate over the 8 bounding box points in the vals lists. Use ints to avoid errors from end floating points
    for i in range(8):
        if(int(z_vals[i]) == int(z_min)):
            z_crop = z_low
        else:
            z_crop = z_high
        z_cropBox.append([x_vals[i], y_vals[i], z_crop])    

    midBox = o3d.geometry.PointCloud()
    midBox.points = o3d.utility.Vector3dVector(z_cropBox)
    midBox_8points = midBox.get_axis_aligned_bounding_box().get_box_points()

    # Iterate over all the points for a given z slice (step size 1)
    # Find and save the lowest y values for each slice
    # Amongst all the minimum y values, find the highest (i.e. the notch apex)
    Y_MINIMA = []
    for i in range(int(z_extent)):
        # Get the slice z-index
        z_slice_min = z_low+i
        z_slice_max = z_low+i+1

        _slice = []
        # iterate over the 8 bounding box points
        for i in range(8):
            if(z_cropBox[i][2] == z_low):
                z_crop = z_slice_min
            else:
                z_crop = z_slice_max
            _slice.append([x_vals[i], y_vals[i], z_crop])


        # crop a single slice 
        sliceBox = o3d.geometry.PointCloud()
        sliceBox.points = o3d.utility.Vector3dVector(np.asarray(_slice))
        crop_zone = o3d.geometry.AxisAlignedBoundingBox.create_from_points(sliceBox.points)
        cropped = bone_pcd.crop(crop_zone) # point cloud right here
        sliceBBox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(sliceBox.points)

        # Get the lowest y value 
        points = np.asarray(cropped.points)
        try:
            if(points.shape[0] != 0):
                Y_MIN = np.amin(points, axis=0)[1]
                corr_z = np.amin(points, axis=0)[2]
                corr_x = np.amin(points, axis=0)[0]
                Y_MINIMA.append((corr_x, Y_MIN, corr_z))
        except:
            pass
            

    # Find the maximum Y val (top of notch)
    NOTCH_ROOF = max(Y_MINIMA, key=itemgetter(0))[1]
    corr_z = max(Y_MINIMA, key=itemgetter(0))[2]
    corr_x = max(Y_MINIMA, key=itemgetter(0))[0]
    # print(f'TARGET= {NOTCH_ROOF} with z={corr_z} x={corr_x}')  

    return (corr_x, NOTCH_ROOF, corr_z)  

def compute_blumensaat(bone_points, notch, TOLERANCE=20):
    """
    Returns x,y (2D) intercondylar notch roof trace
    Can plot x,y as a scatter. Z values possible if 3D trace is needed. 
    """
    bone_pcd = get_pcd(bone_points)
    notch_x = notch[0]
    NOTCH_ROOF = notch[1]

    # Define x,y,z bounds
    x_vals, y_vals, z_vals = get_bounding_box_xyz(bone_points)
    x_min, x_max = get_min_max(x_vals)
    y_min, y_max = get_min_max(y_vals)
    z_min, z_max = get_min_max(z_vals)

    x_extent = x_max-x_min
    y_extent = y_max-y_min
    z_extent = z_max-z_min
    
    # Notch does not go to the edge of the x axis. trim down ??% either side
    x_low = x_min + x_extent*0.05
    x_high = x_max - x_extent*0.2
    x_range = x_high-x_low

    # Notch does not go to the edge of the y axis. trim down 
    y_low = y_min + y_extent*0.20
    y_high = y_max - y_extent*0.50
    y_range = y_high-y_low

    # Set up bounding box to improve point search efficiency

    # Trim z (as before)
    # Set-up z-crop box
    z_low = z_min + z_extent*0.30
    z_high = z_max - z_extent*0.30

    z_cropBox = []
    # iterate over the 8 bounding box points in the vals lists. Use ints to avoid errors from end floating points
    for i in range(8):
        if(int(z_vals[i]) == int(z_min)):
            z_crop = z_low
        else:
            z_crop = z_high
        z_cropBox.append([x_vals[i], y_vals[i], z_crop])    

    midBox = o3d.geometry.PointCloud()
    midBox.points = o3d.utility.Vector3dVector(z_cropBox)
    midBox_8points = midBox.get_axis_aligned_bounding_box().get_box_points()

    # Additional crop of z midBox. 
    x_vals_bline = []
    y_vals_bline = []
    z_vals_bline = []

    for i in midBox_8points:
        x_vals_bline.append(i[0])
        y_vals_bline.append(i[1])
        z_vals_bline.append(i[2])

    x_cropBox = []
    # set-up x/y crop box
    for i in range(8):
        if(x_vals_bline[i] == x_min):
            x_crop = x_low
        if(y_vals_bline[i] == y_min):
            y_crop = y_low
        if(x_vals_bline[i] != x_min):
            x_crop = x_high
        if(y_vals_bline[i] != y_min): 
            y_crop = y_high
        x_cropBox.append([x_crop, y_crop, z_vals_bline[i]])
    # print("NOTCH CROP BOX: ", x_cropBox) 

    # Do the crop
    notchBox = o3d.geometry.PointCloud()
    notchBox.points = o3d.utility.Vector3dVector(x_cropBox)

    boxy = o3d.geometry.AxisAlignedBoundingBox.create_from_points(notchBox.points)
    cropped = bone_pcd.crop(boxy)
    # o3d.visualization.draw_geometries([bone_pcd, boxy])


    # Get a list of the x and y coordinates for the notch line
    bline = []
    for step in range(int(x_range)):
        # Get the slice z-index
        x_slice_min = x_low+step
        x_slice_max = x_low+step+1
        _slice = []
        # iterate over the 8 bounding box points
        for i in range(8):
            if(x_cropBox[i][0]==x_low):
                x_crop = x_slice_min
            else:
                x_crop = x_slice_max
            _slice.append([x_crop, y_vals[i], z_vals[i]])

        # crop a single slice 
        sliceBox = o3d.geometry.PointCloud()
        sliceBox.points = o3d.utility.Vector3dVector(np.asarray(_slice))
        aa_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(sliceBox.points)
        croppedSlice = cropped.crop(aa_box) # point cloud right here
        sliceBBox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(sliceBox.points)

        # Get the max point (track apex of notch)
        points = np.asarray(croppedSlice.points)
        try:
            if(points.shape[0] != 0):
                MAX = np.amax(points, axis=0)[1]
                # Save if the point is near the approx notch apex. 8pts for reglar. Then repeat on failed files with 20.  
                # without the tolerance filter, bones will be rotated incorrectly
                if(MAX < NOTCH_ROOF+10):  #~ adjust tolerance HERE
                    # print([(x_slice_min+x_slice_max)/2, MAX])
                    bline.append([(x_slice_min+x_slice_max)/2, MAX]) # Can also add in z if wanted
                else:
                    # TODO
                    pass
        except Exception as e:
            print('ERROR SAVING POINTS.', e)

    try:
        x,y = np.array(bline).T #~ important x and y of intercondylar line here     
        return x,y
    except Exception as e:
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(exc_type, exc_tb.tb_lineno)

        print(bline)
        # input()
        return e
        # No blumensaat available? Either the knee is rotated wrong... else just use Notch roof from initial. log file. 



def fit_bspline(x,y, plot=False):
    """

    """
    # Fit b-spline
    spline = splrep(x,y, k=3)               # gives (t,c,k). k is degree of spline fit 
    bspline_y_values = splev(x, spline)     # this is for plotting. useless for anything else really 

    bspline_object = BSpline(spline[0], spline[1], spline[2]) # *Important stuff. corresponding Y-evaluation for given x on bspline

    spline_derivatives = spalde(x, spline)

    # Find the max and mins (hills features) on Blumensaat's line
    first = []
    second = [] # unused at the moment. can show whether root is max/min/inflex
    for i in range(len(spline_derivatives[0])): #4
        for d in spline_derivatives:
            if(i==1):
                first.append(d[i])
            elif(i==2):
                second.append(d[i])

    first_deriv_spline = splrep(x, first)

    # Get the roots of the cubic (the hills and notch x values)
    try:
        roots = sproot(first_deriv_spline) #! only seeing 1 root initially 
    except:
        roots = []
    print(f'Roots = {roots}\n')

    #plot
    if(plot):
        plot_blumensaat_profile(x,y, bspline_y_values, bspline_object, spline_derivatives, roots)
    
    return bspline_object, roots


def get_Blumensaats(x, roots, bspline):
    number_of_points = len(roots)
    gradient = 0
    c = 0

    # Get the corresponding y values on the bspline for the root values
    y_coords = []
    for i in roots:
        y_coord = bspline(i).tolist()
        # print(f'y_coord = {y_coord} for x={i}')
        y_coords.append(y_coord)
    
    if(number_of_points >= 3) :
        # ignore middle point?
        gradient = (y_coords[-1]-y_coords[0]) / (roots[-1]-roots[0] )
        c = y_coords[0]-(gradient*roots[0])
    elif(number_of_points == 2):
        # take both
        gradient = (y_coords[1]-y_coords[0]) / (roots[1]-roots[0] )
        c = y_coords[0]-(gradient*roots[0])
    elif(number_of_points == 1):
        # Use notch apex as a reference. Only good if there is a decent estimation of notch roof.....
        # TODO find a different reference. Notch is not always ok but sometimes it's good
        temp_y = bspline(0) # get value where x=0 (central, approx notch)
        gradient = (temp_y-y_coords[0]) / (0-roots[0] )
        c = y_coords[0]-(gradient*roots[0])     
    else:
        print('ERROR: No maxima or minima found on blumensaat\'s line')
        # TODO: Do something useful here. Just cut at the roof?
        temp_y = bspline(0) # get value where x=0 (central, approx notch)
        gradient = 1
        c = 0  
        return -1
    blumensaatLine = gradient*x + c
    print(f'Gradient = {gradient} | c = {c}')
    return blumensaatLine, gradient


def plot_blumensaat_profile(x, y, bspline_y, bspline, spline_derivatives, roots):
    """
    Nice visual plot of x,y (lateral) view of intercondylar notch
    Also draws approximate blumensaat
    """
    # Compute Blumensaat Line from mathematical roots
    y_coords = []
    for i in roots:
        y_coord = bspline(i).tolist()
        print(f'y_coord = {y_coord} for x={i}')
        y_coords.append(y_coord)
    
    blumensaatLine, gradient = get_Blumensaats(x, roots, bspline)

    fig, ax1 = plt.subplots()
    ax1.plot(x, blumensaatLine, color='orange', label='Blumensaat\'s Line')  
    color = 'tab:red'
    ax1.scatter(x,y)
    ax1.set_ylabel('Femur y-coordinate')
    plt.title('Blumensaat\'s Line 2D')
    ax1.set_xlabel('Femur x-coordinate')
    plt.plot(x, bspline_y, label="Lateral Profile of Intercondylar Notch")
    ax2 = ax1.twinx() 
    for i in range(len(spline_derivatives[0])-3):
        ax2.plot(x, [d[i+1] for d in spline_derivatives], '--', label=f"{i+1}st Derivative", color='green')
    axline = ax2.axhline(y=0, color='r', linestyle='-') # straight line at y=0
    ax1.legend(loc=2)
    ax2.legend(loc=1)
    fig.tight_layout() 
    plt.show()


def correction_transform(gradient, bone_points):
    # Get the angle of the line to the x-axis
    angle_in_radians = math.atan(gradient)
    angle_in_degrees = math.degrees(angle_in_radians) # just for debug info. not used elsewhere
    print(f'ANGLE = {angle_in_degrees}')

    # Correction transform 
    theta = angle_in_radians
    cos, sin = np.cos(theta), np.sin(theta)

    # 3D rotation matrix about z-axis. Homogenous coordinates
    R_correction = np.array(((cos, -sin, 0, 0), (sin, cos, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))) 
    print(R_correction)

    # Perform the transform
    bone_homog = np.vstack([bone_points.T, np.ones(bone_points.shape[0])])
    transformed_bone = np.dot(bone_homog.T, R_correction)

    # Drop 4th dimension from affine result
    bone = transformed_bone[:,[0,1,2]]

    # print(R_correction)
    return bone


def correction_footprint(gradient, bone_points):
    """
    Use the gradient to rotate the centroid 
    """
    ROTATION_MATRIX =   [[0.630644, -0.757998, 0.166513, 0.00000000],
                    [-0.756577, -0.648273, -0.0856336, 0.00000000],
                    [0.172856, -0.0719752, -0.982314, 0.00000000],
                    [0.00000000, 0.00000000, 0.00000000, 1.00000000]]

    # Make into homogenous coordinates with a w dimension
    transformed_bone = np.dot(ROTATION_MATRIX, bone_points)

    # Get the angle of the line to the x-axis
    angle_in_radians = math.atan(gradient)

    # Correction transform 
    theta = angle_in_radians
    cos, sin = np.cos(theta), np.sin(theta)

    # 3D rotation matrix about z-axis. Homogenous coordinates
    R_correction = np.array(((cos, -sin, 0, 0), (sin, cos, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))) 

    # Perform the transform
    final_bone = np.dot(R_correction, transformed_bone)

    # Drop 4th dimension from affine result
    bone = transformed_bone[0:3,]
    print(bone)

    return bone


def correction_footprint2(gradient, bone_points):
    """
    Full 3D footprint, not just a centroid 
    """
    ROTATION_MATRIX =   [[0.630644, -0.757998, 0.166513, 0.00000000],
                    [-0.756577, -0.648273, -0.0856336, 0.00000000],
                    [0.172856, -0.0719752, -0.982314, 0.00000000],
                    [0.00000000, 0.00000000, 0.00000000, 1.00000000]]

    # Make homogeonous
    try:
        bone_points = np.vstack([bone_points.T, np.ones(bone_points.shape[0])])
    except:
        print('problem with vstack into homogenous')

    # Perform initial rotation
    try:
        transformed_bone = np.dot(ROTATION_MATRIX, bone_points)
    except Exception as e:
        print(e)

    # Get the angle of the line to the x-axis
    angle_in_radians = math.atan(gradient)

    # Correction transform 
    theta = angle_in_radians
    cos, sin = np.cos(theta), np.sin(theta)

    # 3D rotation matrix about z-axis. Homogenous coordinates
    R_correction = np.array(((cos, -sin, 0, 0), (sin, cos, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))) 

    # Perform the transform
    final_bone = np.dot(R_correction, transformed_bone)

    # Drop 4th dimension from affine result
    bone = transformed_bone[0:3,]
    print(bone)

    return bone


def get_cut_y(bspline, notch_apex):
    """
    """
    x_value = notch_apex[0]
    y_CUT = bspline(x_value)
    return y_CUT


def crop_bone(y_CUT, bone_points):
    """ 
    Crops femur at y value.
    Last step before bounds of footprint can be calculated 
    """
    # apply bounding box
    bone_pcd = o3d.geometry.PointCloud()
    bone_pcd.points = o3d.utility.Vector3dVector(bone_points)
    aa_boundingBox = bone_pcd.get_axis_aligned_bounding_box()
    boundBox = aa_boundingBox.get_extent() 

    # 8 points which define the bounding box
    boundBox_8points = aa_boundingBox.get_box_points()
    # print(np.asarray(boundBox_8points))
    # o3d.visualization.draw_geometries([bone_pcd, aa_boundingBox])

    x_vals = []
    y_vals = []
    z_vals = []

    for i in boundBox_8points:
        x_vals.append(i[0])
        y_vals.append(i[1])
        z_vals.append(i[2])

    y_min = min(y_vals)
    
    # Move the top 4 corners of the bounding box down to Blumensaat's line
    bline_boundingBox_8points = []
    for j in range(8): # it's just iterating over the 8 points of a bounding box.... 
        if(y_vals[j] == y_min): 
            y_cropped = y_vals[j]
        else:
            y_cropped = y_CUT
        bline_boundingBox_8points.append([x_vals[j],y_cropped, z_vals[j]]) 

    cropped_boundBox = o3d.geometry.PointCloud()
    cropped_boundBox.points = o3d.utility.Vector3dVector(bline_boundingBox_8points)
    boxy = o3d.geometry.AxisAlignedBoundingBox.create_from_points(cropped_boundBox.points)

    # Perform the crop
    cropped = bone_pcd.crop(boxy)
    # o3d.visualization.draw_geometries([cropped])
    return cropped


def get_footprint_bounds(bone_points_postcrop, footprint_nodes, bone_points_full, gradient):
    """
    Define bounding box extents for use as denominators / reference frame for % method analysis
    @Input: 
    @Output: [c/C min, c/C max, n/N min, n/N max]
    # TODO add new bound in lateral to medial plane. l/L min and l/L max
    """
    # Get new bounding box for the cropped figure
    postCropBBox = bone_points_postcrop.get_axis_aligned_bounding_box() # ~ This here is the main reference frame for relative footprint position calculations
    postCropBBox_8points = np.asarray(postCropBBox.get_box_points())

    # Display the cropped section and it's bounding box
    # o3d.visualization.draw_geometries([bone_points_postcrop, postCropBBox])

    x_vals = postCropBBox_8points[:,0]
    y_vals = postCropBBox_8points[:,1]
    z_vals = postCropBBox_8points[:,2]

    # Get X and Y ranges. Ignore Z component when looking at lateral view.
    y_min = min(y_vals)
    y_max = max(y_vals)
    y_range = y_max - y_min             # 100% n/N
    assert(abs(y_range) != 0)

    x_min = min(x_vals)
    x_max = max(x_vals)
    x_range = x_max - x_min             # 100% c/C
    assert(abs(x_range) != 0)

    # Z isn't even used in the calculation but this is here for completion's sake
    # TODO: make z into the l/L bounding number
    z_min = min(z_vals)
    z_max = max(z_vals)
    z_range = z_max - z_min             # 100% l/L
    assert(abs(z_range) != 0)


    ###############################################################
    # Get footprint edges as a relative % of extent
    ###############################################################
    # footprint_coords = footprint_nodes.T
    # # print('footprint Coords: ' , footprint_coords)

    # Get the bounds of the footprint in the x and y directions.
    # Convert Node numbers --> local coords
    footprint_coords = bone_points_full[footprint_nodes]
    if(len(footprint_coords) <= 3):
        footprint_x = fxMin = fxMax = footprint_coords[0]
        footprint_y = fyMin = fyMax = footprint_coords[1]
        footprint_z = fzMin = fzMax = footprint_coords[2]
    else:
        footprint_x = footprint_coords[:,0]
        footprint_y = footprint_coords[:,1]
        footprint_z = footprint_coords[:,2]
    

        # # Get the bounds
        fxMin = min(footprint_x)
        fxMax = max(footprint_x)

        fyMin = min(footprint_y)
        fyMax = max(footprint_y)

        fzMin = min(footprint_z)
        fzMax = max(footprint_z)

    # % Condyle depth
    minCD = (x_max - fxMax) / x_range
    maxCD = (x_max - fxMin) / x_range
    print('************************')
    print(f'c/C Bounds: {minCD}, {maxCD}')

    # % intercondylar notch height
    minNH = (y_max - fyMax) / y_range
    maxNH = (y_max - fyMin) / y_range

    # Correct to zero if it's at/over the edge. 
    if (minNH < 0):
        minNH = 0

    print(f'n/N Bounds: {minNH}, {maxNH}')

    # % lateral-medial bone width
    minlL = (z_max - fzMax) / z_range
    maxlL = (z_max - fzMin) / z_range
    print(f'l/L Bounds: {minlL}, {maxlL}')

    print('************************')

    # Save into the bounds dictionary for writing to the csv file
    data = [minCD, maxCD, minNH, maxNH, minlL, maxlL]
    return data


def write_to_CSV(path, bounds):
    """
    Write the bounds to a .csv
    @Input: save path as string
            bounds as a dictionary
    """
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerows(bounds.items())
    print('File written successfully')