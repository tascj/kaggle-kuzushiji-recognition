import random
import torch
import numpy as np
from tqdm import tqdm

import mmcv
from mmdet.core.bbox import bbox_overlaps


def np_iof(bboxes, canvas):
    return bbox_overlaps(
        torch.FloatTensor(bboxes),
        torch.FloatTensor(canvas),
        'iof',
    ).numpy().reshape(-1)


def iter_empty_crops(w, h, stride, size):
    xs = range(0, w - size + 1, stride)
    xs = sorted(set(list(xs) + [w - size]))
    ys = range(0, h - size + 1, stride)
    ys = sorted(set(list(xs) + [h - size]))
    for x in xs:
        for y in ys:
            yield [x, y, x + size, y + size]


def iter_target_crops(bboxes, w, h, size):
    available_bboxes = bboxes.copy()
    while len(available_bboxes) > 0:
        # random pick a bbox
        idx = np.random.randint(0, len(available_bboxes))
        bbox = available_bboxes[idx]
        # random pick a crop region including the selected bbox
        x1, y1, x2, y2 = bbox
        lt_region = np.array([x2 - size, y2 - size, x1, y1])
        lt_region[0::2] = lt_region[0::2].clip(0, w - size)
        lt_region[1::2] = lt_region[1::2].clip(0, h - size)
        assert (lt_region[2] >= lt_region[0]) and \
                (lt_region[3] >= lt_region[1])

        # make a crop
        x = np.random.randint(lt_region[0], lt_region[2] + 1)
        y = np.random.randint(lt_region[1], lt_region[3] + 1)
        crop = np.array([[x, y, x + size, y + size]])

        inds = np_iof(available_bboxes, crop) > 0.8
        available_bboxes = available_bboxes[~inds]
        yield crop.flatten().tolist()


def crop_gt(bboxes, labels, x1, y1, x2, y2):
    crop = np.array([[x1, y1, x2, y2]])
    iof = np_iof(bboxes, crop)
    inds = iof > 0.8
    ret_labels = labels[inds].copy()
    ret_bboxes = bboxes[inds].copy()
    ret_bboxes -= np.array([x1, y1, x1, y1])
    ret_bboxes[:, 0::2] = ret_bboxes[:, 0::2].clip(0, x2 - x1)
    ret_bboxes[:, 1::2] = ret_bboxes[:, 1::2].clip(0, y2 - y1)
    return ret_bboxes, ret_labels


def main():
    random.seed(0)
    np.random.seed(0)
    dtrainval = mmcv.load('../data/dtrainval.pkl')
    dtrainval_crop = []
    SIZE = 1024
    LABEL_DUMMY = 4788
    crops = []
    for sample in tqdm(dtrainval):
        img = mmcv.imread('../data/train_images/' + sample['filename'])
        w, h = sample['width'], sample['height']
        bboxes = sample['ann']['bboxes']
        labels = sample['ann']['labels']
        idx_crop = 0
        base_name = sample['filename'].rstrip('.jpg')
        if len(bboxes) == 0:
            for x1, y1, x2, y2 in iter_empty_crops(w, h, SIZE - 64, SIZE):
                img_crop = img[y1:y2, x1:x2]
                img_crop[8:24, 8:24] = img_crop[8:24, 8:24] * 0.5 + [0, 127, 0]
                bboxes_crop = np.array([[8, 8, 24, 24]], np.float32)
                labels_crop = np.array([LABEL_DUMMY], np.int64)

                filename = '{}_{}.png'.format(base_name, idx_crop)
                mmcv.imwrite(
                    img_crop,
                    '../data/train_crops/' + filename,
                    auto_mkdir=True)
                dtrainval_crop.append({
                    'filename': filename,
                    'width': SIZE,
                    'height': SIZE,
                    'ann': {
                        'bboxes':
                        bboxes_crop.reshape(-1, 4),
                        'labels':
                        labels_crop.reshape(-1),
                        'bboxes_ignore':
                        np.array([], dtype=np.float32).reshape(-1, 4),
                        'labels_ignore':
                        np.array([], dtype=np.int64).reshape(-1, )
                    },
                })
                idx_crop += 1

        else:
            for x1, y1, x2, y2 in iter_target_crops(bboxes, w, h, SIZE):
                img_crop = img[y1:y2, x1:x2]
                bboxes_crop, labels_crop = \
                        crop_gt(bboxes, labels, x1, y1, x2, y2)

                filename = '{}_{}.png'.format(base_name, idx_crop)
                mmcv.imwrite(
                    img_crop,
                    '../data/train_crops/' + filename,
                    auto_mkdir=True)
                dtrainval_crop.append({
                    'filename': filename,
                    'width': SIZE,
                    'height': SIZE,
                    'ann': {
                        'bboxes':
                        bboxes_crop.reshape(-1, 4),
                        'labels':
                        labels_crop.reshape(-1),
                        'bboxes_ignore':
                        np.array([], dtype=np.float32).reshape(-1, 4),
                        'labels_ignore':
                        np.array([], dtype=np.int64).reshape(-1, )
                    },
                })
                idx_crop += 1

    mmcv.dump(
        [im for im in dtrainval_crop if not im['filename'].startswith('umgy')],
        '../data/dtrain_crop.pkl')
    mmcv.dump(dtrainval_crop, '../data/dtrainval_crop.pkl')


if __name__ == "__main__":
    main()
