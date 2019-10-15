import argparse
import mmcv
import pandas as pd
import numpy as np
from tqdm import tqdm
from mmdet.ops import nms


def iter_results(pred):
    for idx, bboxes in enumerate(pred):
        for x1, y1, x2, y2, p in bboxes:
            yield idx, int((x1 + x2) / 2), int((y1 + y2) / 2)


def post_process(preds, num_classes=4787, iou_thr=0.3, score_thr=0.3):
    ret = []
    for pred in tqdm(preds):
        bboxes = np.vstack(pred)
        labels = np.concatenate([[i] * len(bb) for i, bb in enumerate(pred)])
        # nms
        _, inds = nms(bboxes, iou_thr)
        bboxes, labels = bboxes[inds], labels[inds]
        # score filtering
        inds = bboxes[:, 4] > score_thr
        bboxes, labels = bboxes[inds], labels[inds]
        #
        ret.append([bboxes[labels == i] for i in range(num_classes)])
    return ret


def merge(model_preds):
    if len(model_preds) == 0:
        return model_preds[0]
    ret = model_preds[0]
    for preds in model_preds[1:]:
        for idx, sample_pred in enumerate(preds):
            ret[idx] = [
                np.vstack([bb0, bb1])
                for bb0, bb1 in zip(ret[idx], sample_pred)
            ]
    return ret


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=str)
    parser.add_argument('--input', type=str, nargs='+')
    parser.add_argument('--iou-thr', type=float, default=0.3)
    parser.add_argument('--score-thr', type=float, default=0.3)
    return parser.parse_args()


def main():
    args = parse_args()
    unicode_translation = pd.read_csv('../data/unicode_translation.csv')
    class2unicode = dict(
        zip(unicode_translation.index.values, unicode_translation['Unicode']))

    sub = pd.read_csv('../data/sample_submission.csv')
    sub = sub.set_index('image_id')

    metas = mmcv.load('../data/dtest.pkl')
    model_preds = [mmcv.load(input_path) for input_path in args.input]
    assert all(len(metas) == len(preds) for preds in model_preds)
    preds = merge(model_preds)
    preds = post_process(preds, iou_thr=args.iou_thr, score_thr=args.score_thr)
    for meta, pred in tqdm(zip(metas, preds), total=len(preds)):
        image_id = meta['filename'].rstrip('.jpg')
        labels = []
        for idx, x, y in iter_results(pred):
            unicode = class2unicode[idx]
            labels.append('{} {} {}'.format(unicode, x, y))
        labels = ' '.join(labels)
        sub.loc[image_id, 'labels'] = labels
    sub = sub.reset_index()

    sub.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
