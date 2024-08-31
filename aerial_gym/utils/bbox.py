import torch
import cv2


def seg_pics2bbox(seg_pics, obj_idx):
    """
    Input a batch of segmentation pictures and the index of target object, return the bbox
    If the target object is not in the segmentation area, return (0, 0, 0, 0)

    Input: seg_pics(batch_size, w, h), obj_idx(int)
    Output:bbox vector(batch_size, x, y, w, h)
    """
    xs = []
    ys = []
    hs = []
    ws = []
    batch_size = seg_pics.size(0)


    for i in range(batch_size):
        non_zero_indices = (seg_pics[i] == obj_idx).nonzero(as_tuple=True)
        
        if len(non_zero_indices[0]) > 0:
            min_row = non_zero_indices[0].min().item()
            max_row = non_zero_indices[0].max().item()

            min_col = non_zero_indices[1].min().item()
            max_col = non_zero_indices[1].max().item()
            
            h = max_row - min_row
            w = max_col - min_col
            x = min_row + h / 2
            y = min_col + h / 2
        else:
            x = y = 0
            h = w = 0

        hs.append(h)
        ws.append(w)
        xs.append(x)
        ys.append(y)

    hs = torch.tensor(hs)
    ws = torch.tensor(ws)
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)

    output = torch.stack([xs, ys, ws, hs], dim=1)

def hand_pic2bbox(pic):
    """
    Input a picture, a window will come out. Manually select a bbox.

    Input: np(w, h, bgr)
    Output: (x, y, w, h)
    """
    cv2.namedWindow('tmp', cv2.WND_PROP_FULLSCREEN)
    output = cv2.selectROI('tmp', pic, False, False)
    return output