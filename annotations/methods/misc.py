IMG_DIR = 'D:\\Datasets\\Surveillance\\obj_1\\'
NEW_DIR = "E:\\original\\"

import os
from pathlib import Path

names = [str(d)[(str(d).rfind("\\") + 1):] for d in Path(IMG_DIR).glob("*W00RD*") if "_annotated.jpeg" in str(d) or "_colour.jpeg" in str(d)]
moving_names = [d for d in names if int(d[:6]) >= 11146]

import shutil

for moving_name in moving_names:
    old_path = IMG_DIR + moving_name
    new_path = NEW_DIR + moving_name
    shutil.copy(old_path, new_path)


#   C: Move files between directories with given regex
'''

def GetP(camera):

    W = camera["screenW"]
    H = camera["screenH"]
    vfov = np.radians(camera["vfov"])
    fclip = camera["fclip"]
    nclip = camera["nclip"]

    P = np.array([[H/(W*tan(vfov/2)), 0, 0, 0], 
                  [0, 1/tan(vfov/2), 0, 0], 
                  [0, 0, -(nclip + fclip)/(nclip - fclip), -fclip*nclip/(nclip - fclip)], 
                  [0, 0, 1, 0]])

    return P

def W2S(V, P, screenW, screenH, coords):

    ncol = coords.shape[1]
    coords = np.vstack((coords, [1] * ncol))
    clip = np.dot(P, np.dot(V, coords))
    ndc = clip/clip[3, :]

    T = np.array([[screenW/2, 0, 0, screenW/2],
                  [0, -screenH/2, 0, screenH/2]])

    res = np.dot(T, ndc).astype(int)
    return res

'''