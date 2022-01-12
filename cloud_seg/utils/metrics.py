import numpy as np

def intersection_and_union(pred, true, pcut=0.5):
    """
    Calculates intersection and union for an image.
    Args:
        pred (np.array): an array predictions
        true (torc.Tensor): a tensor of labels
    Returns:
        intersection (int): total intersection of pixels
        union (int): total union of pixels
    """

    # Intersection and union totals 
    pred_flattened = ((pred > 0.5) * 1).flatten()
    true_flattened = true.flatten()

    intersection = np.logical_and(true_flattened, pred_flattened)/pred_flattened.shape[0]
    union = np.logical_or(true_flattened, pred_flattened)/pred_flattened.shape[0]

    return float(np.sum(intersection)), float(np.sum(union))
