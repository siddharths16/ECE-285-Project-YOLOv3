import argparse
import json

from torch.utils.data import DataLoader

from AAmodels import *
from utils.datasets import *
from utils.utils import *

import pdb
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import cv2


def test(cfg,
         data,
         weights=None,
         batch_size=16,
         img_size=416,
         iou_thres=0.5,
         conf_thres=0.001,
         nms_thres=0.5,
         save_json=False,
         model=None, AAshift=0, bpf=0):
    # Initialize/load model and set device
    if model is None:
        device = torch_utils.select_device(opt.device)
        verbose = True

        # Initialize model
        model = Darknet(cfg, img_size, bpf=bpf).to(device)
        # Remove once done with cls plot
        hyp = {'giou': 3.31,  # giou loss gain
               'cls': 42.4,  # cls loss gain
               'cls_pw': 1.0,  # cls BCELoss positive_weight
               'obj': 40.0,  # obj loss gain
               'obj_pw': 1.0,  # obj BCELoss positive_weight
               'iou_t': 0.213,  # iou training threshold
               'lr0': 0.00261,  # initial learning rate (SGD=1E-3, Adam=9E-5)
               'lrf': -4.,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
               'momentum': 0.949,  # SGD momentum
               'weight_decay': 0.000489,  # optimizer weight decay
               'fl_gamma': 0.5,  # focal loss gamma
               'hsv_h': 0.0103,  # image HSV-Hue augmentation (fraction)
               'hsv_s': 0.691,  # image HSV-Saturation augmentation (fraction)
               'hsv_v': 0.433,  # image HSV-Value augmentation (fraction)
               'degrees': 1.43,  # image rotation (+/- deg)
               'translate': 0.0663,  # image translation (+/- fraction)
               'scale': 0.11,  # image scale (+/- gain)
               'shear': 0.384}  # image shear (+/- deg)
        model.hyp = hyp
        model.arc = 'default'
        # Note for Rohit

        # Load weights
        attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(
                weights, map_location=device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(model, weights)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:
        device = next(model.parameters()).device  # get model device
        verbose = False

    # Configure run
    data = parse_data_cfg(data)
    nc = int(data['classes'])  # number of classes
    model.nc = nc
    test_path = data['valid']  # path to test images
    names = load_classes(data['names'])  # class names

    # Dataloader
    dataset = LoadImagesAndLabels(test_path, img_size, batch_size)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=min([os.cpu_count(), batch_size, 16]),
                            pin_memory=True,
                            collate_fn=dataset.collate_fn)

    if AAshift > 0:
        print("In AAShift routine in test.py")
        p_mat = list()
        r_mat = list()
        f1_mat = list()
        mp_mat = list()
        mr_mat = list()
        map_mat = list()
        mf1_mat = list()
        loss_mat = list()
        for AAsh in range(AAshift+1):
            print(f"Shift of {AAsh}")
            seen = 0
            model.eval()
            coco91class = coco80_to_coco91_class()
            s = ('%20s' + '%10s' * 6) % ('Class', 'Images',
                                         'Targets', 'P', 'R', 'mAP', 'F1')
            p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0.
            loss = torch.zeros(3)
            jdict, stats, ap, ap_class = [], [], [], []
            print('Reinitialize variables')
            for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
                # if batch_i<309:
                #     continue
                trans_mat = np.float32([[1, 0, AAsh], [0, 1, AAsh]])
                # pdb.set_trace()
                for inum, img in enumerate(imgs):
                    tcoords = targets[targets[:, 0] == inum, :]
                    # pdb.set_trace()
                    for ti, tcoord in enumerate(tcoords):
                        topright_xy = [tcoord[2]+tcoord[4]/2+AAsh /
                                       img_size, tcoord[3]+tcoord[5]/2+AAsh/img_size]
                        if all(coords <= 1 for coords in topright_xy):
                            # print("In warpAffine loop")
                            # Changing the x coordinate of bbox
                            tcoords[ti][2] += AAsh/img_size
                            # Changing the y coordinate of bbox
                            tcoords[ti][3] += AAsh / img_size
                        else:
                            break
                        imgs[inum] = torch.roll(imgs[inum], shifts=(
                            0, AAsh, AAsh), dims=(0, 1, 2))
                    # print(f"Successfully warped image {inum+1} in batch {batch_i}")
                    targets[targets[:, 0] == inum, :] = tcoords
                targets = targets.to(device)
                imgs = imgs.to(device)
                _, _, height, width = imgs.shape  # batch size, channels, height, width

                # Plot images with bounding boxes
                if batch_i == 0 and not os.path.exists('test_batch0.jpg'):
                    plot_images(imgs=imgs, targets=targets,
                                paths=paths, fname='test_batch0.jpg')

                # Run model
                # inference and training outputs
                inf_out, train_out = model(imgs)
                # pdb.set_trace()
                # Compute loss
                # print(model.nc)
                # pdb.set_trace()
                if hasattr(model, 'hyp'):  # if model has loss hyperparameters
                    # GIoU, obj, cls
                    loss += compute_loss(train_out, targets,
                                         model)[1][:3].cpu()
                # print(f"Loss: {loss}")
                # Run NMS
                output = non_max_suppression(
                    inf_out, conf_thres=conf_thres, nms_thres=nms_thres)

                # Statistics per image
                for si, pred in enumerate(output):
                    labels = targets[targets[:, 0] == si, 1:]
                    nl = len(labels)
                    tcls = labels[:, 0].tolist() if nl else []  # target class
                    seen += 1

                    if pred is None:
                        if nl:
                            stats.append(
                                ([], torch.Tensor(), torch.Tensor(), tcls))
                        continue

                    # Append to text file
                    # with open('test.txt', 'a') as file:
                    #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]

                    # Append to pycocotools JSON dictionary
                    if save_json:
                        # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                        image_id = int(Path(paths[si]).stem.split('_')[-1])
                        box = pred[:, :4].clone()  # xyxy
                        scale_coords(imgs[si].shape[1:], box,
                                     shapes[si])  # to original shape
                        box = xyxy2xywh(box)  # xywh
                        # xy center to top-left corner
                        box[:, :2] -= box[:, 2:] / 2
                        for di, d in enumerate(pred):
                            jdict.append({'image_id': image_id,
                                          'category_id': coco91class[int(d[6])],
                                          'bbox': [floatn(x, 3) for x in box[di]],
                                          'score': floatn(d[4], 5)})

                    # Clip boxes to image bounds
                    clip_coords(pred, (height, width))

                    # Assign all predictions as incorrect
                    correct = [0] * len(pred)
                    if nl:
                        detected = []
                        tcls_tensor = labels[:, 0]

                        # target boxes
                        tbox = xywh2xyxy(labels[:, 1:5])
                        tbox[:, [0, 2]] *= width
                        tbox[:, [1, 3]] *= height

                        # Search for correct predictions
                        for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred):

                            # Break if all targets already located in image
                            if len(detected) == nl:
                                break

                            # Continue if predicted class not among image classes
                            if pcls.item() not in tcls:
                                continue

                            # Best iou, index between pred and targets
                            m = (pcls == tcls_tensor).nonzero().view(-1)
                            iou, bi = bbox_iou(pbox, tbox[m]).max(0)

                            # If iou > threshold and class is correct mark as correct
                            # and pcls == tcls[bi]:
                            if iou > iou_thres and m[bi] not in detected:
                                correct[i] = 1
                                detected.append(m[bi])

                    # Append statistics (correct, conf, pcls, tcls)
                    stats.append(
                        (correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls))
            # pdb.set_trace()
            # Compute statistics
            stats = [np.concatenate(x, 0)
                     for x in list(zip(*stats))]  # to numpy
            # pdb.set_trace()
            if len(stats):
                p, r, ap, f1, ap_class = ap_per_class(*stats)
                mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
                # number of targets per class
                nt = np.bincount(stats[3].astype(np.int64), minlength=nc)
            else:
                nt = torch.zeros(1)
            # Print results
            pf = '%20s' + '%10.3g' * 6  # print format
            # pdb.set_trace()
            print(f"With AAsh as {AAsh}")
            # pdb.set_trace()
            print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))
            # p, r, f1, mp, mr, map, mf1
            # pdb.set_trace()
            p_mat.append(p)
            r_mat.append(r)
            f1_mat.append(f1)
            mp_mat.append(mp)
            mr_mat.append(mr)
            map_mat.append(map)
            mf1_mat.append(mf1)
            # *(loss / len(dataloader)).tolist())
            loss_mat.append((loss / len(dataloader)).numpy())
            # Print results per class
            if verbose and nc > 1 and len(stats):
                for i, c in enumerate(ap_class):
                    print(pf % (names[c], seen, nt[c],
                                p[i], r[i], ap[i], f1[i]))

            # Save JSON
            if save_json and map and len(jdict):
                try:
                    imgIds = [int(Path(x).stem.split('_')[-1])
                              for x in dataset.img_files]
                    with open('results.json', 'w') as file:
                        json.dump(jdict, file)

                    from pycocotools.coco import COCO
                    from pycocotools.cocoeval import COCOeval

                    # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                    # initialize COCO ground truth api
                    cocoGt = COCO('../coco/annotations/instances_val2014.json')
                    # initialize COCO pred api
                    cocoDt = cocoGt.loadRes('results.json')

                    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
                    # [:32]  # only evaluate these images
                    cocoEval.params.imgIds = imgIds
                    cocoEval.evaluate()
                    cocoEval.accumulate()
                    cocoEval.summarize()
                    map = cocoEval.stats[1]  # update mAP to pycocotools mAP
                except:
                    print(
                        'WARNING: missing dependency pycocotools from requirements.txt. Can not compute official COCO mAP.')

    else:
        seen = 0
        model.eval()
        coco91class = coco80_to_coco91_class()
        s = ('%20s' + '%10s' * 6) % ('Class', 'Images',
                                     'Targets', 'P', 'R', 'mAP', 'F1')
        p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0.
        loss = torch.zeros(3)
        jdict, stats, ap, ap_class = [], [], [], []
        for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
            targets = targets.to(device)
            imgs = imgs.to(device)
            _, _, height, width = imgs.shape  # batch size, channels, height, width

            # Plot images with bounding boxes
            if batch_i == 0 and not os.path.exists('test_batch0.jpg'):
                plot_images(imgs=imgs, targets=targets,
                            paths=paths, fname='test_batch0.jpg')

            # Run model
            inf_out, train_out = model(imgs)  # inference and training outputs
            # pdb.set_trace()
            # Compute loss
            if hasattr(model, 'hyp'):  # if model has loss hyperparameters
                loss += compute_loss(train_out, targets,
                                     model)[1][:3].cpu()  # GIoU, obj, cls
            # print(f"Loss: {loss}")
            # pdb.set_trace()
            # Run NMS
            output = non_max_suppression(
                inf_out, conf_thres=conf_thres, nms_thres=nms_thres)

            # Statistics per image
            for si, pred in enumerate(output):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                seen += 1

                if pred is None:
                    if nl:
                        stats.append(
                            ([], torch.Tensor(), torch.Tensor(), tcls))
                    continue

                # Append to text file
                # with open('test.txt', 'a') as file:
                #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]

                # Append to pycocotools JSON dictionary
                if save_json:
                    # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                    image_id = int(Path(paths[si]).stem.split('_')[-1])
                    box = pred[:, :4].clone()  # xyxy
                    scale_coords(imgs[si].shape[1:], box,
                                 shapes[si])  # to original shape
                    box = xyxy2xywh(box)  # xywh
                    # xy center to top-left corner
                    box[:, :2] -= box[:, 2:] / 2
                    for di, d in enumerate(pred):
                        jdict.append({'image_id': image_id,
                                      'category_id': coco91class[int(d[6])],
                                      'bbox': [floatn(x, 3) for x in box[di]],
                                      'score': floatn(d[4], 5)})

                # Clip boxes to image bounds
                clip_coords(pred, (height, width))

                # Assign all predictions as incorrect
                correct = [0] * len(pred)
                if nl:
                    detected = []
                    tcls_tensor = labels[:, 0]

                    # target boxes
                    tbox = xywh2xyxy(labels[:, 1:5])
                    tbox[:, [0, 2]] *= width
                    tbox[:, [1, 3]] *= height

                    # Search for correct predictions
                    for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred):

                        # Break if all targets already located in image
                        if len(detected) == nl:
                            break

                        # Continue if predicted class not among image classes
                        if pcls.item() not in tcls:
                            continue

                        # Best iou, index between pred and targets
                        m = (pcls == tcls_tensor).nonzero().view(-1)
                        iou, bi = bbox_iou(pbox, tbox[m]).max(0)

                        # If iou > threshold and class is correct mark as correct
                        # and pcls == tcls[bi]:
                        if iou > iou_thres and m[bi] not in detected:
                            correct[i] = 1
                            detected.append(m[bi])

                # Append statistics (correct, conf, pcls, tcls)
                stats.append(
                    (correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls))

        # Compute statistics
        stats = [np.concatenate(x, 0) for x in list(zip(*stats))]  # to numpy
        if len(stats):
            p, r, ap, f1, ap_class = ap_per_class(*stats)
            mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
            # number of targets per class
            nt = np.bincount(stats[3].astype(np.int64), minlength=nc)
        else:
            nt = torch.zeros(1)

        # Print results
        pf = '%20s' + '%10.3g' * 6  # print format
        print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))

        # Print results per class
        if verbose and nc > 1 and len(stats):
            for i, c in enumerate(ap_class):
                print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

        # Save JSON
        if save_json and map and len(jdict):
            try:
                imgIds = [int(Path(x).stem.split('_')[-1])
                          for x in dataset.img_files]
                with open('results.json', 'w') as file:
                    json.dump(jdict, file)

                from pycocotools.coco import COCO
                from pycocotools.cocoeval import COCOeval

                # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                # initialize COCO ground truth api
                cocoGt = COCO('../coco/annotations/instances_val2014.json')
                # initialize COCO pred api
                cocoDt = cocoGt.loadRes('results.json')

                cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
                # [:32]  # only evaluate these images
                cocoEval.params.imgIds = imgIds
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                map = cocoEval.stats[1]  # update mAP to pycocotools mAP
            except:
                print(
                    'WARNING: missing dependency pycocotools from requirements.txt. Can not compute official COCO mAP.')

    if AAshift == 0:
        # Return results
        maps = np.zeros(nc) + map
        for i, c in enumerate(ap_class):
            maps[c] = ap[i]
        return (mp, mr, map, mf1, *(loss / len(dataloader)).tolist()), maps
    else:
        # pdb.set_trace()
        maps_mat = list()
        for i in range(len(p_mat)):
            maps = np.zeros(nc) + map
            for i, c in enumerate(ap_class):
                maps[c] = ap[i]
            maps_mat.append(maps.mean())
        return(mp_mat, mr_mat, map_mat, mf1_mat, loss_mat), maps_mat


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--cfg', type=str,
                        default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data', type=str,
                        default='data/coco.data', help='coco.data file path')
    parser.add_argument('--weights', type=str,
                        default='weights/yolov3-spp.weights', help='path to weights file')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=416,
                        help='inference size (pixels)')
    parser.add_argument('--iou-thres', type=float, default=0.5,
                        help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float,
                        default=0.001, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5,
                        help='iou threshold for non-maximum suppression')
    parser.add_argument('--save-json', action='store_true',
                        help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--device', default='',
                        help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--test_AA_shift', type=int, default=0,
                        help='Test shift operation on AA model')
    parser.add_argument('--AAmode', type=int, default=0,
                        help='Control anti aliasing')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        results_mat, maps_mat = test(opt.cfg,
                                     opt.data, weights=opt.weights,
                                     batch_size=opt.batch_size, img_size=opt.img_size, iou_thres=opt.iou_thres, conf_thres=opt.conf_thres, nms_thres=opt.nms_thres, save_json=opt.save_json, AAshift=opt.test_AA_shift, bpf=opt.AAmode)
        AA_file_name = f'AA_shift_cls_{opt.AAmode}.jpg'
        AA_text_file = f'AA_shift_cls_{opt.AAmode}.txt'

        maps_mat = results_mat[-1][-1][-1]
        pdb.set_trace()
        fig = plt.figure(figsize=(10, 10))
        x = list(range(opt.test_AA_shift+1))
        plt.plot(x, maps_mat)
        plt.title('Shift Invariance Test', fontsize=12)
        plt.xlabel('Shift (in pixels)', fontsize=16)
        plt.xticks(x, x, fontsize=20)
        plt.yticks(fontsize=18)
        plt.ylabel('Classification Loss', fontsize=20)
        plt.ylabel('Cls', fontsize=20)
        fig.tight_layout()
        if not os.path.exists(AA_file_name):
            fig.savefig(AA_file_name, dpi=200)

        with open(AA_text_file, 'a') as fh:
            fh.write(f"{x}\n")
            fh.write(f"{maps_mat}\n")
        plt.close()
