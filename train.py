from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse
import cv2
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from tensorboardX import SummaryWriter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--learning_rate", default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=12, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3-voc0712.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/voc0712.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, default='./weights/darknet53.conv.74',help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=True, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    opt = parser.parse_args()
    #print("type(opt) --> ",type(opt))
    print(opt)
    
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initiate model
    model = Darknet(opt.model_def,img_size=opt.img_size).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)
    
    for i,p in enumerate(model.named_parameters()):
        if i == 156:
            break
        p[1].requires_grad = False
        
    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    
    # Get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    writer = SummaryWriter('logs')
    optimizer = torch.optim.Adam(model.parameters(),lr=opt.learning_rate)    
    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]
    count_show = 30 #每训练30步显示一次训练效果
    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            if batches_done % count_show:
                current_dim = int(imgs.shape[-1])
                with torch.no_grad():
                    detections = non_max_suppression(outputs, opt.conf_thres, opt.nms_thres)
                    imgs_show_list = []
                    for count in range(len(detections)):
                        if detections[count] is not None:
                            imgs_show = cv2.imread(_[count])
                            imgs_show = cv2.cvtColor(imgs_show,cv2.COLOR_BGR2RGB)
                            orig_h = imgs_show.shape[0]
                            orig_w = imgs_show.shape[1]
                            # The amount of padding that was added
                            pad_x = max(orig_h - orig_w, 0) * (current_dim / max(orig_h,orig_w))
                            pad_y = max(orig_w - orig_h, 0) * (current_dim / max(orig_h,orig_w))
                            # Image height and width after padding is removed
                            unpad_h = current_dim - pad_y
                            unpad_w = current_dim - pad_x
                            for i in detections[count]:
                                i = i.numpy()
                                x1 = int((i[0] - pad_x // 2) / unpad_w * orig_w)
                                y1 = int((i[1] - pad_y // 2) / unpad_h * orig_h)
                                x2 = int((i[2] - pad_x // 2) / unpad_w * orig_w)
                                y2 = int((i[3] - pad_y // 2) / unpad_h * orig_h)
                                cv2.rectangle(imgs_show,(x1,y1),(x2,y2),(0,255,255),3)
                                cv2.putText(imgs_show,class_names[int(i[-1])],(int(x1+20),int(y1+20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)       
                            imgs_show = torch.from_numpy(imgs_show).permute(2,0,1).float()
                            imgs_show,pad = pad_to_square(imgs_show,0)
                            imgs_show = F.interpolate(imgs_show.unsqueeze(0), size=current_dim, mode="nearest").squeeze(0)
                            imgs_show = imgs_show[:,:,:]/255.0
                            imgs_show_list.append(imgs_show)
                    if len(imgs_show_list) > 0:
                        imgs_show_pt = torch.stack([img for img in imgs_show_list])  
                    # imgs_pt = torch.from_numpy(imgs_show).permute(2,0,1).squeeze(0)
                        imgs_grid = torchvision.utils.make_grid(imgs_show_pt)    
                    else:
                        imgs_grid = torchvision.utils.make_grid(imgs)
                    writer.add_image('eight_voc_images',imgs_grid)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------
            
            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                for targe, value in tensorboard_log:
                    writer.add_scalar(targe, value, batches_done)
                #logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)

        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            for targe, value in evaluation_metrics:
                writer.add_scalar(targe, value, epoch)
            #logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
    writer.close()
