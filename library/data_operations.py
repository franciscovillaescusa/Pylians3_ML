import numpy as np
import sys, os, time

# This function takes a 3D density field and performs a rotation and/or flip
# df --------> 3D density field
# rot -------> rotation (an integer number from 0 to 23)
# flip ------> whether flip the cube (0 or 1)
def rotation_flip_3D(df, rot, flip):
    
    #np.rot(field, number of 90deg rotations, axes)
    if   rot==0:   df_new = np.rot90(df, 1, (0,1)) 
    elif rot==1:   df_new = np.rot90(df, 2, (0,1))
    elif rot==2:   df_new = np.rot90(df, 3, (0,1))
    elif rot==3:   df_new = np.rot90(df, 1, (0,2)) 
    elif rot==4:   df_new = np.rot90(df, 2, (0,2))
    elif rot==5:   df_new = np.rot90(df, 3, (0,2))
    elif rot==6:   df_new = np.rot90(df, 1, (1,2)) 
    elif rot==7:   df_new = np.rot90(df, 2, (1,2))   
    elif rot==8:   df_new = np.rot90(df, 3, (1,2))
    elif rot==9:   df_new = np.rot90(np.rot90(df, 2, (0,2)), 1, (0,1)) 
    elif rot==10:  df_new = np.rot90(np.rot90(df, 2, (1,2)), 1, (0,1)) 
    elif rot==11:  df_new = np.rot90(np.rot90(df, 3, (1,2)), 1, (0,1)) 
    elif rot==12:  df_new = np.rot90(np.rot90(df, 3, (0,2)), 1, (0,1)) 
    elif rot==13:  df_new = np.rot90(np.rot90(df, 2, (0,1)), 1, (0,2)) 
    elif rot==14:  df_new = np.rot90(np.rot90(df, 2, (1,2)), 1, (0,2))
    elif rot==15:  df_new = np.rot90(np.rot90(df, 3, (0,1)), 1, (0,2)) 
    elif rot==16:  df_new = np.rot90(np.rot90(df, 3, (1,2)), 1, (0,2))
    elif rot==17:  df_new = np.rot90(np.rot90(df, 2, (0,1)), 1, (1,2))
    elif rot==19:  df_new = np.rot90(np.rot90(df, 2, (0,2)), 1, (1,2)) 
    elif rot==18:  df_new = np.rot90(np.rot90(df, 3, (0,1)), 1, (1,2))
    elif rot==20:  df_new = np.rot90(np.rot90(df, 3, (0,2)), 1, (1,2)) 
    elif rot==21:  df_new = np.rot90(np.rot90(df, 1, (0,1)), 1, (1,2))
    elif rot==22:  df_new = np.rot90(np.rot90(df, 3, (0,2)), 3, (1,2))
    elif rot==23:  df_new = df
    
    # do fliping
    if flip==1:  df_new = np.flip(df_new, axis=-1)

    return df_new


# This routine tests whether all rotations/flippping produce unique fields
# grid -----> will generate a random 3D field with (grid,grid,grid) dimensions
def test_rotation_flip_3D(grid):

    cubes = 48
    df0 = np.random.random((grid,grid,grid))
    df  = np.zeros((cubes,grid,grid,grid))

    # rotate/flip the cube in all 48 possible ways
    for i in range(cubes):
        df[i] = rotation_flip_3D(df0, i%24, i//24)

    # check if two cubes are the same
    for i in range(cubes):
        for j in range(i+1,cubes):
            if np.array_equal(df[i],df[j]):  
                raise Exception('Density fields %d and %d are the same!!!'%(i,j))

    print('All %d cubes are different :)'%cubes)
