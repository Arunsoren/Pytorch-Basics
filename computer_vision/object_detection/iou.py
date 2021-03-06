import torch

def intersection_over_union(boxes_pred, boxes_labels, box_format="midpoint"):
    # boxes_preds shape is (N, 4) where N is the number of BBoxes
    # boxes_labels shape is (N, 4)
    #returns Intersection_over_union

    if box_format == "midpoint": #yolo format
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2


    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4] #(N, 1) #slice just to get the second dim

        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    #intersection points
    x1 = torch.max(box1_x1, box2_x1)  #max(x1, X1)
    y1 = torch.max(box1_y1, box2_y1)  #max(y1, Y1)
    x2 = torch.min(box1_x2, box2_x2)  #min(x2, X2)
    y2 = torch.min(box1_y2, box2_y2)  #min(y2, Y2) 

    #intersection .clamp(0) when no intersection
    intersection = (x2-x1).clamp(0) * (y2-y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y1 -box1_y2))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y1 - box2_y2))

    return intersection / (box1_area + box2_area - intersection + 1e-6)






































