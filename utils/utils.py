import torch
import torch.nn as nn
from collections import Counter

class YoloLoss(nn.Module):
    """
    Calculate the loss for YoloV1 model
    """
    def __init__(self, S=7, B=2, C=3):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        """
        S is split size of image (grid size) (in paper 7)
        B is number of boxes (in paper 2),
        C is number of classes (in paper 20, in dataset 3)
        """
        self.S = S
        self.B = B
        self.C = C

        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # input prediction shape: (BATCH_SIZE, S*S*(C+B*5))
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        #Calculate IOU for two predicted bounding boxes with target box
        iou_b1 = intersection_over_union(predictions[...,self.C + 1: self.C+5], target[..., self.C + 1: self.C + 5])
        iou_b2 = intersection_over_union(predictions[...,self.C + 6: self.C+10], target[..., self.C + 1: self.C + 5])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        #Take the box with highest IOU out of the two predictions. Note: bestbox will be indices 0,1 for which bbox was best
        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., self.C].unsqueeze(3)

        # =========================#
        # FOR BOX COORDINATES      #
        # =========================#

        #set boxes with no object in them to 0. We only take out one of the two predictions, which is one with highest IOUs
        box_predictions = exists_box * ((bestbox * predictions[..., self.C + 6: self.C+10]+ (1- bestbox)* predictions[..., self.C + 1: self.C + 5]))
        box_targets = exists_box * target[..., self.C + 1: self.C+ 5]

        # take sqrt of width and height of box
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4])* torch.sqrt(torch.abs(box_predictions[...,2:4]+ 1e-6))
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(torch.flatten(box_predictions, end_dim=-2),
                            torch.flatten(box_targets, end_dim=-2))

        # =========================#
        # FOR  OBJECT LOSS      #
        # =========================#

        #pred_box is the confidence score for bbox with highest IOU
        pred_box = (bestbox * predictions[..., self.C + 5:self.C+6] + (1-bestbox)* predictions[..., self.C:self.C+1])
        object_loss = self.mse(torch.flatten(exists_box* pred_box),
                                torch.flatten(exists_box * target[..., self.C: self.C+1]))


        # =========================#
        # FOR  No OBJECT LOSS      #
        # =========================#

        no_object_loss = self.mse(torch.flatten((1-exists_box)* predictions[..., self.C:self.C+1], start_dim=1),
                                torch.flatten((1-exists_box)* target[..., self.C:self.C+1], start_dim=1))

        no_object_loss += self.mse(torch.flatten((1-exists_box)* predictions[..., self.C+5:self.C+6], start_dim=1),
                                torch.flatten((1-exists_box)* target[..., self.C:self.C+1], start_dim=1))

        # =========================#
        # FOR  CLASS LOSS      #
        # =========================#

        class_loss = self.mse(torch.flatten(exists_box * predictions[..., :self.C], end_dim=-2),
                            torch.flatten(exists_box * target[..., :self.C], end_dim=-2))

        loss = (self.lambda_coord * box_loss + object_loss + self.lambda_noobj * no_object_loss + class_loss)

        return loss

def intersection_over_union(boxes_preds, boxes_labels, box_format = 'midpoint'):
    """
    Calculates Intersection over union

    Parameters:
        boxes_preds (tensor): Prediction of Bounding boxes(BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes are (x,y,w,h) or (x1,y1,x2,y2) respectively

    Returns:
        tensor: Intersection over union for all examples
    """
    # boxes_preds shape is (N,4) where N is the number of predicted bboxes
    # boxes_labels shape is (n, 4)

    if box_format == "midpoint":
        box1_x1 = boxes_preds[...,0:1] - boxes_preds[...,2:3] / 2
        box1_y1 = boxes_preds[...,1:2] - boxes_preds[...,3:4] / 2
        box1_x2 = boxes_preds[...,0:1] + boxes_preds[...,2:3] / 2
        box1_y2 = boxes_preds[...,1:2] + boxes_preds[...,3:4] / 2

        box2_x1 = boxes_labels[...,0:1] - boxes_labels[...,2:3] / 2
        box2_y1 = boxes_labels[...,1:2] - boxes_labels[...,3:4] / 2
        box2_x2 = boxes_labels[...,0:1] + boxes_labels[...,2:3] / 2
        box2_y2 = boxes_labels[...,1:2] + boxes_labels[...,3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[...,0:1]
        box1_y1 = boxes_preds[...,1:2]
        box1_x2 = boxes_preds[...,2:3]
        box1_y2 = boxes_preds[...,3:4] #output tensor should (N,1). If we only use 3, we go to (N)

        box2_x1 = boxes_labels[...,0:1]
        box2_y1 = boxes_labels[...,1:2]
        box2_x2 = boxes_labels[...,2:3]
        box2_y2 = boxes_labels[...,3:4]


    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they don't intersect. Since when they don't intersect one of these will be negetive so they should become 0
    intersection = (x2 - x1).clamp(0) * (y2-y1).clamp(0)

    box1_area = abs((box1_x2- box1_x1)*(box1_y2-box1_y1))
    box2_area = abs((box2_x2- box2_x1)*(box2_y2-box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_supression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Given Bounding Boxes does Non-maximal supression

    Parameters:
        bboxes (list): list of lists containing all bounding boxes with each box specified as [class_preds, prob_score, x1, y2, x2, y2]
        iou_threshold: (float): threshold where predicted bounding box is correct
        threshold (float): threshold to remove predicted bboxes (Independent of IOU)
        box_format (str): midpoint/corners, if boxes are (x,y,w,h) or (x1,y1,x2,y2) respectively

    Returns:
        bboxes_after_nms (list): bboxes after performing NMS given a specific IOU Threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1]> threshold ]
    bboxes = sorted(bboxes, key= lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [box for box in bboxes
        if box[0]!=chosen_box[0] or intersection_over_union(torch.tensor(chosen_box[2:]), torch.tensor(box[2:]), box_format = box_format) < iou_threshold
        ]
        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20):
    """
    Calculates Mean Average Precision

    Parameters:
        pred_boxes (list): list of lists containing all bounding boxes with each box specified as [train_idx, class_preds, prob_score, x1, y1, x2, y2]
        true_boxes (list): similar to pred boxes except all the correct ones
        iou_threshold (float): threshold above which predicted box is correct
        box_format (str): midpoint/corners, if boxes are (x,y,w,h) or (x1,y1,x2,y2) respectively
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IOU threshold

    """

    #list storing all AP for respective classes
    average_precisions = []

    #used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all the predictions and targets and only add the ones that belong to the current class c
        for detection in pred_boxes:
            if detection[1]==c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1]==c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example, Counter here finds how many ground truth boxes we get for each training example.
        # so lets say img 0 has 3 and img 1 has 5 then we will get dictionary with amount_bboxes={0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary and convert to the following (w.r.t. same example)
        #  amount_bboxes = {0: torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box_probabilities which is index 2
        detections.sort(key= lambda x:x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # if none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # only take out ground truths that have the same training_idx as detection
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(torch.tensor(detection[3:]), torch.tensor(gt[3:]), box_format= box_format)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx]==0:
                    #true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            # if iou is lower then the detection is false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1], dtype=precisions.dtype), precisions))
        recalls = torch.cat((torch.tensor([0], dtype=precisions.dtype), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions)/ len(average_precisions)

def get_bboxes(loader, model, iou_threshold, threshold, pred_format = "cells", box_format = "midpoint", device= "cuda"):

    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval() mode before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_supression(bboxes[idx],iou_threshold= iou_threshold, threshold = threshold, box_format = box_format)

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx]+ nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1]>threshold:
                    all_true_boxes.append([train_idx]+box)

            train_idx+=1

    model.train()
    return all_pred_boxes, all_true_boxes

def convert_cellboxes(predictions, S=7, C=3):
    """
    Converts bounding boxes output from Yolo with an image split size of S into entire image ratios rather than relative to cell ratios.
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, C + 10)
    bboxes1 = predictions[..., C+1:C+5]
    bboxes2 = predictions[..., C+6: C+10]
    scores = torch.cat((predictions[...,C].unsqueeze(0), predictions[...,C+5].unsqueeze(0)), dim=0)
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1-best_box)+ best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1]+ cell_indices)
    y = 1 / S * (best_boxes[..., 1:2]+ cell_indices.permute(0,2,1,3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x,y,w_y), dim=-1)
    predicted_class = predictions[...,:C].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., C], predictions[...,C+5]).unsqueeze(-1)
    converted_preds = torch.cat((predicted_class, best_confidence, converted_bboxes), dim=-1)

    return converted_preds

def cellboxes_to_boxes(out, S=7):
    converted_preds = convert_cellboxes(out).reshape(out.shape[0], S*S, -1)
    converted_preds[...,0] = converted_preds[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S*S):
            bboxes.append([x.item() for x in converted_preds[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

def save_checkpoint(state, filename = "my_checkpoint.pth"):
    print("=> Saving Checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])