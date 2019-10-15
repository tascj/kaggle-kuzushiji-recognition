import mmcv
import numpy as np
from terminaltables import AsciiTable

from .bbox_overlaps import bbox_overlaps
from .class_names import get_classes


def tpfp_default(det_bboxes, gt_bboxes, gt_ignore, iou_thr, area_ranges=None):
    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)
    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            det_areas = (det_bboxes[:, 2] - det_bboxes[:, 0] + 1) * (
                det_bboxes[:, 3] - det_bboxes[:, 1] + 1)
            for i, (min_area, max_area) in enumerate(area_ranges):
                fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
        return tp, fp
    if len(det_bboxes) > 0:
        xc = (det_bboxes[:, 2] + det_bboxes[:, 0]) / 2
        yc = (det_bboxes[:, 3] + det_bboxes[:, 1]) / 2
        _det_bboxes = det_bboxes.copy()
        _det_bboxes[:, 0] = xc
        _det_bboxes[:, 1] = yc
        _det_bboxes[:, 2] = xc + 1
        _det_bboxes[:, 3] = yc + 1
        det_bboxes = _det_bboxes
        # pass
    ious = bbox_overlaps(det_bboxes, gt_bboxes)
    ious_max = ious.max(axis=1)
    ious_argmax = ious.argmax(axis=1)
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore, dtype=bool)
        else:
            gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (
                gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)
            gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not (gt_ignore[matched_gt] or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                    else:
                        fp[k, i] = 1
                # otherwise ignore this detected bbox, tp = 0, fp = 0
            elif min_area is None:
                fp[k, i] = 1
            else:
                bbox = det_bboxes[i, :4]
                area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    return tp, fp


def get_cls_results(det_results, gt_bboxes, gt_labels, gt_ignore, class_id):
    """Get det results and gt information of a certain class."""
    cls_dets = [det[class_id]
                for det in det_results]  # det bboxes of this class
    cls_gts = []  # gt bboxes of this class
    cls_gt_ignore = []
    for j in range(len(gt_bboxes)):
        gt_bbox = gt_bboxes[j]
        cls_inds = (gt_labels[j] == class_id + 1)
        cls_gt = gt_bbox[cls_inds, :] if gt_bbox.shape[0] > 0 else gt_bbox
        cls_gts.append(cls_gt)
        if gt_ignore is None:
            cls_gt_ignore.append(np.zeros(cls_gt.shape[0], dtype=np.int32))
        else:
            cls_gt_ignore.append(gt_ignore[j][cls_inds])
    return cls_dets, cls_gts, cls_gt_ignore


def eval_f1(det_results,
            gt_bboxes,
            gt_labels,
            gt_ignore=None,
            scale_ranges=None,
            iou_thr=0.,
            dataset=None,
            print_summary=True):
    """
    Dirty F1.
    """
    assert len(det_results) == len(gt_bboxes) == len(gt_labels)
    if gt_ignore is not None:
        assert len(gt_ignore) == len(gt_labels)
        for i in range(len(gt_ignore)):
            assert len(gt_labels[i]) == len(gt_ignore[i])
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    eval_results = []
    num_classes = len(det_results[0])  # positive class num
    gt_labels = [
        label if label.ndim == 1 else label[:, 0] for label in gt_labels
    ]
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gt_ignore = get_cls_results(
            det_results, gt_bboxes, gt_labels, gt_ignore, i)
        # calculate tp and fp for each image
        tpfp_func = tpfp_default
        tpfp = [
            tpfp_func(cls_dets[j], cls_gts[j], cls_gt_ignore[j], iou_thr,
                      area_ranges) for j in range(len(cls_dets))
        ]
        tp, fp = tuple(zip(*tpfp))
        # calculate gt number of each scale, gts ignored or beyond scale
        # are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for j, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += np.sum(np.logical_not(cls_gt_ignore[j]))
            else:
                gt_areas = (bbox[:, 2] - bbox[:, 0] + 1) * (
                    bbox[:, 3] - bbox[:, 1] + 1)
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum(
                        np.logical_not(cls_gt_ignore[j])
                        & (gt_areas >= min_area) & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'tp': tp,
            'fp': fp,
        })
    results = eval_results
    tp = sum(z['tp'][0, -1] if z['tp'].shape[1] > 0 else 0 for z in results)
    num_gts = sum(z['num_gts'] for z in results)
    num_dets = sum(z['num_dets'] for z in results)
    recall = tp / (num_gts + eps)
    precision = tp / (num_dets + eps)
    f1_score = 2 * recall * precision / (recall + precision + eps)
    header = ['gts', 'dets', 'recall', 'precision', 'F1']
    metrics = [
        num_gts,
        num_dets,
        '{:.4f}'.format(recall),
        '{:.4f}'.format(precision),
        '{:.4f}'.format(f1_score),
    ]
    table_data = [header, metrics]
    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    print(table.table)
    return f1_score, table_data
