import numpy as np

def crop_face(img, left, top, right, bottom):

    scale_factor = 0.8

    ## Calculate center points and rectangle side length
    width = right - left
    height = bottom - top
    cX = left + width // 2
    cY = top + height // 2
    M = (abs(width) + abs(height)) / 2

    ## Get the resized rectangle points
    newLeft = max(0, int(cX - scale_factor * M))
    newTop = max(0, int(cY - scale_factor * M))
    newRight = min(img.shape[1], int(cX + scale_factor * M))
    newBottom = min(img.shape[0], int(cY + scale_factor * M))

    ## Draw the circle and bounding boxes
    face_crop = np.copy(img[newTop:newBottom, newLeft:newRight])
    return face_crop
