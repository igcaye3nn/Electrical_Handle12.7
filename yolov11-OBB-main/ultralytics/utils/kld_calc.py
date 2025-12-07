# prompt: Divide the "final function" (none, sqrt, ln) into 3 cases, as an input argument & returning value
import torch
import math

def getklddist(box_pr, box_gt, final_function='none', eps=1e-7, final_reLU=False):
  """
  Calculates the KLD-based loss between predicted and ground truth rotated bounding boxes
  with an option to apply a final function to KLD.

  Args:
    box_pr (torch.Tensor): Predicted bounding boxes (N, 5) in (xc, yc, w, h, theta) format.
    box_gt (torch.Tensor): Ground truth bounding boxes (N, 5) in (xc, yc, w, h, theta) format.
    final_function (str): Specifies the final function to apply to KLD.
                           'none': No function applied.
                           'sqrt': Square root of KLD.
                           'ln': Natural logarithm of KLD + 1.
    eps (float): A small epsilon value to prevent division by zero or log(0).

  Returns:
    tuple: A tuple containing:
      - loss (torch.Tensor): Loss calculated based on the specified final_function.
      - KLD (torch.Tensor): The Kullback-Leibler Divergence values.
  """
  #datatype conversion from float16 to float32
  original_dtype = box_pr.dtype
  box_pr = box_pr.float()
  box_gt = box_gt.float()

  # Ensure positive dimensions for safety
  pr_w, pr_h = box_pr[:, 2].clamp(min=eps), box_pr[:, 3].clamp(min=eps)
  gt_w, gt_h = box_gt[:, 2].clamp(min=eps), box_gt[:, 3].clamp(min=eps)

  # Extract elements for calculation
  gt_xc, gt_yc, gt_theta = box_gt[:, 0], box_gt[:, 1], box_gt[:, 4]
  pr_xc, pr_yc, pr_theta = box_pr[:, 0], box_pr[:, 1], box_pr[:, 4]

  # Calculate delta_theta
  delta_theta = pr_theta - gt_theta

  # Calculate diff_x and diff_y
  diff_x = pr_xc - gt_xc
  diff_y = pr_yc - gt_yc

  # Calculate the first part of the alternative KL-term
  part1 = (4 / (gt_w**2 + eps)) * torch.square((diff_x * torch.cos(gt_theta)) + (diff_y * torch.sin(gt_theta)))

  # Calculate the second part of the alternative KL-term
  part2 = (4 / (gt_h**2 + eps)) * torch.square((diff_y * torch.cos(gt_theta)) - (diff_x * torch.sin(gt_theta)))

  # Calculate the alternative 1st KL-term
  alternative_kl_term1 = part1 + part2

  # Calculate the terms for the alternative 2nd KL-term
  term1 = ((pr_h**2 + eps) / (gt_w**2 + eps)) * torch.square(torch.sin(delta_theta))
  term2 = ((pr_w**2 + eps) / (gt_h**2 + eps)) * torch.square(torch.sin(delta_theta))
  term3 = ((pr_h**2 + eps) / (gt_h**2 + eps)) * torch.square(torch.cos(delta_theta))
  term4 = ((pr_w**2 + eps) / (gt_w**2 + eps)) * torch.square(torch.cos(delta_theta))
  term5 = torch.log((gt_h**2 + eps) / (pr_h**2 + eps))
  term6 = torch.log((gt_w**2 + eps) / (pr_w**2 + eps))

  # Calculate the alternative 2nd KL-term
  alternative_kl_term2 = term1 + term2 + term3 + term4
  alternative_kl_term3 = term5 + term6

  # Calculate the final KLD value (clamp to avoid negative values due to numerical instability)
  KLD = 0.5 * alternative_kl_term1 + 0.5 * alternative_kl_term2 + 0.5 * alternative_kl_term3 - 1.0
  KLD = torch.clamp(KLD, min=eps)
  KLD = torch.relu(KLD) # Ensure KLD is non-negative

  # Apply the final function based on the argument
  if final_function == 'none':
    transformed_KLD = KLD
  elif final_function == 'sqrt':
    transformed_KLD = torch.sqrt(KLD)
  elif final_function == 'ln':
    transformed_KLD = torch.log(KLD + 1 + eps)
  elif final_function == 'exp':
    transformed_KLD = 1 - torch.exp(KLD)
  elif final_function == 'neg_exp':
    transformed_KLD = 1 - torch.exp((-1)*(KLD))
  else:
    raise ValueError("Invalid final_function specified. Choose 'none', 'sqrt', 'ln', 'exp', or 'neg_exp'.")

  # Set Tau
  Tau = 1

  # Calculate the loss using the transformed KLD
  loss = 1 - (1 / (Tau + transformed_KLD + eps))

  if final_reLU:
      loss = torch.relu(loss)
  loss = loss.to(original_dtype)
  KLD = KLD.to(original_dtype)

  return loss, KLD

def getlostkld(transformed_KLD,Tau=1.0, eps=1e-7):
  loss = 1 - (1 / (Tau + transformed_KLD + eps))
  return loss


# Example Usage (using the original tensor definitions for demonstration)
if __name__ == "__main__":
  box_pr = torch.tensor([
          [100.0, 100.0, 50.0, 30.0, 0.0],
          [200.0, 200.0, 60.0, 40.0, math.pi/4],
          [300.0, 300.0, 70.0, 50.0, math.pi/2]
      ])

  box_gt = torch.tensor([
          [110.0, 100.0, 50.0, 30.0, 0.0],
          [205.0, 195.0, 60.0, 40.0, math.pi/4],
          [300.0, 300.0, 70.0, 50.0, math.pi/2]
      ])

  # Calculate loss using 'none' final function
  loss_none, KLD_none = getklddist(box_pr, box_gt, final_function='none')
  print("KLD values (none):")
  print(KLD_none)
  print("Loss (none):")
  print(loss_none)

  # Calculate loss using 'sqrt' final function
  loss_sqrt, KLD_sqrt = getklddist(box_pr, box_gt, final_function='sqrt')
  print("\nKLD values (sqrt):")
  print(KLD_sqrt)
  print("Loss (sqrt):")
  print(loss_sqrt)

  # Calculate loss using 'ln' final function
  loss_log, KLD_log = getklddist(box_pr, box_gt, final_function='ln')
  print("\nKLD values (ln):")
  print(KLD_log)
  print("Loss (ln):")
  print(loss_log)

  # Calculate loss using 'exp' final function
  loss_exp, KLD_exp = getklddist(box_pr, box_gt, final_function='exp')
  print("\nKLD values (exp):")
  print(KLD_exp)
  print("Loss (exp):")
  print(loss_exp)

  # Calculate loss using 'inv_exp' final function
  loss_inv_exp, KLD_inv_exp = getklddist(box_pr, box_gt, final_function='neg_exp')
  print("\nKLD values (neg_exp):")
  print(KLD_inv_exp)
  print("Loss (neg_exp):")
  print(loss_inv_exp)
