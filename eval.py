def Maskgen_model(outputs, ind):
    out = outputs["instances"][outputs["instances"].pred_classes == ind].pred_masks.to('cpu')
    out = out.numpy()
    return(out)

def Maskgen(image, color_code):
    mask = (image == color_code).all(-1)
    return(mask)


def Metrics(gt,pred):
    intersection = np.logical_and(gt,pred)
    union = np.logical_or(gt,pred)
    IOU = np.count_nonzero(intersection) / np.count_nonzero(union)
    Dice_coeff = np.count_nonzero(intersection) / (np.count_nonzero(gt) + np.count_nonzero(pred))
    return((IOU,Dice_coeff))
