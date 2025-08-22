# Bounding box utilities functions

def is_inside(inner_box, outer_box):
    """
    Check if the center of the inner bounding box lies inside the outer bounding box.
    
    Args:
        inner_box (tuple): Coordinates of the inner box (x1, y1, x2, y2).
        outer_box (tuple): Coordinates of the outer box (x1, y1, x2, y2).
    
    Returns:
        bool: True if the center of the inner box is inside the outer box, False otherwise.
    """
    # Unpack coordinates
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box
    
    # Calculate the center of the inner box
    cx = (ix1 + ix2) // 2
    cy = (iy1 + iy2) // 2
    
    # Check if center lies inside the outer box
    return ox1 <= cx <= ox2 and oy1 <= cy <= oy2


def compress_box(box, compression=0.05):
    """
    Shrinks the bounding box inward by a given compression ratio on all sides.
    
    Args:
        box (tuple): Coordinates of the box (x1, y1, x2, y2).
        compression (float): Fraction of width/height to reduce from each side (default 0.05 = 5%).
    
    Returns:
        tuple: Compressed bounding box coordinates (new_x1, new_y1, new_x2, new_y2).
    """
    # Unpack box coordinates
    x1, y1, x2, y2 = box
    
    # Calculate width and height
    w, h = x2 - x1, y2 - y1
    
    # Apply compression on each side
    new_x1 = int(x1 + compression * w)
    new_y1 = int(y1 + compression * h)
    new_x2 = int(x2 - compression * w)
    new_y2 = int(y2 - compression * h)
    
    return new_x1, new_y1, new_x2, new_y2


def iou(boxA, boxB):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        boxA (tuple): Coordinates of the first box (x1, y1, x2, y2).
        boxB (tuple): Coordinates of the second box (x1, y1, x2, y2).
    
    Returns:
        float: IoU value in the range [0, 1].
    """
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # Compute intersection area
    interArea = max(0, xB - xA) * max(0, yB - yA)
    
    # Compute areas of both boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    # Avoid division by zero
    if boxAArea + boxBArea - interArea == 0:
        return 0
    
    # Compute IoU = Intersection / Union
    return interArea / float(boxAArea + boxBArea - interArea)
