from IPython import get_ipython
from IPython.display import display
# %%
import numpy as np
import torch
import math

def bbox_reformat(bbox, format_type):
  """
  Reformats a bounding box from one format to another.

  Args:
    bbox: A NumPy array representing a bounding box.
    format_type: A string indicating the desired output format.
                 Currently supports 'xyxy' for [x1, y1, x2, y2].

  Returns:
    A NumPy array representing the bounding box in the specified format.
  """
  if format_type == 'xyxy':
    # Assuming input format is [center_x, center_y, width, height, angle]
    # This reformatting is for axis-aligned bounding boxes from center-based format.
    # For oriented bounding boxes, converting to xyxy [x1, y1, x2, y2]
    # is not a direct conversion and usually requires considering the rotation.
    # This is a placeholder for axis-aligned conversion.
    center_x, center_y, width, height, angle = bbox
    x1 = center_x - width / 2
    y1 = center_y - height / 2
    x2 = center_x + width / 2
    y2 = center_y + height / 2
    return torch.Tensor([x1, y1, x2, y2])
  else:
    raise ValueError(f"Unsupported format type: {format_type}")
# %%
def calculate_iou_batch(obb_gt_matrix, obb_pr_matrix, eps=1e-7, CIoU=False):
  """
  Calculates the Intersection over Union (IoU) between multiple oriented bounding boxes.

  Args:
    obb_gt_matrix: A NumPy array representing ground truth oriented bounding boxes.
                   Format is [N x [center_x, center_y, width, height, angle]].
    obb_pr_matrix: A NumPy array representing predicted oriented bounding boxes.
                   Format is [N x [center_x, center_y, width, height, angle]].
    eps: A small epsilon value to avoid division by zero.
    CIoU: A boolean indicating whether to calculate Complete IoU (CIoU).
          (Note: CIoU calculation is not implemented in this basic function)

  Returns:
    A NumPy array containing the calculated IoU values for each pair of bounding boxes.
  """
  if obb_gt_matrix.shape != obb_pr_matrix.shape:
    raise ValueError("Input matrices must have the same shape.")

  #If the input matrices are torch tensors, convert to tensor.

  num_boxes = obb_gt_matrix.shape[0]
  iou_values = np.zeros(num_boxes)
  w1mat = np.zeros(num_boxes)
  h1mat = np.zeros(num_boxes)
  w2mat = np.zeros(num_boxes)
  h2mat = np.zeros(num_boxes)

  for i in range(num_boxes):
    obb_gt = obb_gt_matrix[i]
    obb_pr = obb_pr_matrix[i]

    #check format
    if obb_gt.size == 4 or 5: # Assuming 5 values mean [xc, yc, w, h, theta] for oriented box corners
      obb_gt_reformatted = bbox_reformat(obb_gt, 'xyxy') # Example call, adjust as needed
      obb_pr_reformatted = bbox_reformat(obb_pr, 'xyxy') # Example call, adjust as needed
      pass
    else: # Assuming 8 values mean [x1, y1, x2, y2, x3, y3, x4, y4] for oriented box corners
      obb_gt_reformatted = np.array(box1[0], box1[1], box1[2], box1[3])
      obb_gt_reformatted = np.array(box2[0], box2[1], box2[2], box2[3])

    # Convert reformatted NumPy arrays to PyTorch tensors
    #b1_x1, b1_y1, b1_x2, b1_y2 = torch.tensor(obb_gt_reformatted)
    #b2_x1, b2_y1, b2_x2, b2_y2 = torch.tensor(obb_pr_reformatted)
    b1_x1, b1_y1, b1_x2, b1_y2 = obb_gt_reformatted.clone().detach()
    b2_x1, b2_y1, b2_x2, b2_y2 = obb_pr_reformatted.clone().detach()

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    #print(inter)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    #print(union)

    # IoU calculation (assuming axis-aligned boxes after reformatting)
    iou = inter / union
    iou_values[i] = iou.item() # Convert PyTorch tensor to Python number
    #print(iou_values[i])

  iou_tensor = torch.tensor(iou_values).unsqueeze(1)
  #print(iou_tensor)

  if CIoU:  # only include the wh aspect ratio part
    wa1, ha1 = obb_gt_matrix[..., 2:4].split(1, dim=-1)
    wa2, ha2 = obb_pr_matrix[..., 2:4].split(1, dim=-1)
    v = (4 / math.pi**2) * ((wa2 / ha2).atan() - (wa1 / ha1).atan()).pow(2)
    with torch.no_grad():
      alpha = v / (v - iou_tensor + (1 + eps))
      CIOU = iou_tensor - v * alpha  # CIoU
      #print(CIOU)
    return CIOU.cuda()

  else:
    return iou_tensor.cuda()

# %%
if __name__ == '__main__':
    # Example input matrices (3x5)
    obb_gt_example = torch.tensor([
    [100, 100, 50, 30, math.radians(45)],  # Box 1: center(100,100), w=50, h=30, angle=45 degrees
    [200, 200, 40, 40, math.radians(0)],   # Box 2: center(200,200), w=40, h=40, angle=0 degrees
    [50, 150, 60, 25, math.radians(-30)]   # Box 3: center(50,150), w=60, h=25, angle=-30 degrees
    ])

    obb_pr_example = torch.tensor([
    [105, 105, 55, 35, math.radians(40)],  # Predicted Box 1
    [195, 195, 45, 45, math.radians(5)],   # Predicted Box 2
    [55, 145, 65, 30, math.radians(-25)]   # Predicted Box 3
    ])

    # Call the function with example inputs
    iou_results = calculate_iou_batch(obb_gt_example, obb_pr_example, eps=1e-7)
    print(iou_results)
    CIOU = calculate_iou_batch(obb_gt_example, obb_pr_example, eps=1e-7, CIoU=True)
    print(CIOU)
    iou_diff = iou_results - CIOU
    print(iou_diff)
    # To see the result, print iou_results
# %%
