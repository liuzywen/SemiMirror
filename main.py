import argparse
import torch.optim
from torch import nn

from util.utils import *
import yaml
from pathlib import Path
from model.model_helper import ModelBuilder
from util.loss_opr import ProbOhemCrossEntropy2d, CriterionOhem, compute_unsupervised_loss_by_threshold
import torch.nn.functional as F
from dataset.augmentation import  cut_mix_label_adaptive,cut_mix_label_adaptive_ori,cut_mix_label_adaptive_strong
from datetime import datetime
from util.loss import compute_L_UL_loss,StructLoss
from model.lr_learning.lr_helper import *
import os
from model.lr_learning.lr_helper import consistency_weight

best_mae = 1
best_epoch = 0
best_student_mae = 1
best_student_epoch = 0

def get_opt(model, LR, MOMENTUM, WD):
    params = [
        {
            "params": [
                p for name, p in model.named_parameters()
                if ("bias" in name or "bn" in name)
            ],
            "weight_decay":
                0,
        },
        {
            "params": [
                p for name, p in model.named_parameters()
                if ("bias" not in name and "bn" not in name)
            ]
        },
    ]
    optimizer = torch.optim.SGD(
        params,
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WD,
    )
    return optimizer


def train(cfg):
    '''创建save_path和logging'''
    save_path = Path(cfg["saver"]["snapshot_dir"])
    save_path.mkdir(exist_ok=True)
    logger, write = config_log(save_path, tensorboard=False)
    logger.info("Train: save checkpoint in : {}".format(str(save_path)))

    '''创建模型'''
    model = ModelBuilder(cfg["net"])
    modules_back = [model.encoder_d, model.encoder_r]
    modules_head = [model.decoder, model.decoder_r, model.decoder_d]
    model.cuda()

    '''创建数据集'''
    lab_loader, unlab_loader, test_loader = get_model_and_dataloader(cfg)

    '''创建优化器和损失函数'''
    if cfg['criterion']['type'] == 'CELoss': #使用交叉熵损失函数
        criterion_sup = nn.CrossEntropyLoss(ignore_index=255)
        criterion_unsup = nn.CrossEntropyLoss(ignore_index=255)
    elif cfg['criterion']['type'] == 'ProbOhemCrossEntropy2d':
        pixel_num = 50000 * cfg['dataset']['batch_size']
        criterion_sup = ProbOhemCrossEntropy2d(ignore_label=255, thresh=0.5, min_kept=pixel_num, use_weight=True)
        criterion_unsup = ProbOhemCrossEntropy2d(ignore_label=255, thresh=0.5, min_kept=pixel_num, use_weight=True)
    elif cfg['criterion']['type'] == 'BCEloss':
        criterion_sup = nn.BCEWithLogitsLoss(reduction='mean')
        criterion_unsup = nn.MSELoss(reduction='mean')
    elif cfg['criterion']['type'] == 'struloss':
        criterion_sup = StructLoss()
        criterion_unsup = StructLoss()
    elif cfg['criterion']['type'] == 'ohem':
        criterion_sup = CriterionOhem(aux_weight=0, thresh = 0.7, min_kept= 100000)
        criterion_unsup = CriterionOhem(aux_weight=0, thresh = 0.7, min_kept= 100000)

    MSELoss = torch.nn.MSELoss()
    '''损失权重优化器'''
    rampup_ends = int(cfg['trainer']['ramp_up'] * cfg['trainer']['epochs'])
    cons_w_unsup = consistency_weight(final_w=cfg['trainer']['unsupervised']['unsupervised_w'], iters_per_epoch=len(unlab_loader),rampup_starts=1,
                                      rampup_ends=rampup_ends)

    cfg_trainer = cfg["trainer"]
    cfg_optim = cfg_trainer["optimizer"]
    params_list = []
    for module in modules_back:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"])
        )
    for module in modules_head:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"])
        )

    optimizer_kwargs = cfg['trainer']['optimizer']['kwargs']
    if cfg['trainer']['optimizer']['type'] == 'AdamW':
        optimizer = torch.optim.AdamW(params_list, **optimizer_kwargs)
    else:
        optimizer = get_opt(model, **optimizer_kwargs)

    '''教师模型'''
    model_teacher = ModelBuilder(cfg["net"])
    model_teacher = model_teacher.cuda()
    for p in model_teacher.parameters():
        p.requires_grad = False

    '''开始训练'''
    global best_epoch,best_mae,best_student_mae, best_student_epoch
    last_epoch = 1
    if cfg["saver"].get('auto_resume',False):
        lastest_model = os.path.join(cfg["saver"]["snapshot_dir"], 'ckpt.pth')
        if not os.path.exists(lastest_model):
            "No checkpoint found in '{}'".format(lastest_model)
        else:
            print(f"Resume model from:'{lastest_model}'")
            checkpoint_resume = torch.load(lastest_model)
            model.load_state_dict(checkpoint_resume['model_state'], strict=True)
            optimizer.load_state_dict(checkpoint_resume['optimizer_state'])
            model_teacher.load_state_dict(checkpoint_resume['teacher_state'])
            last_epoch = checkpoint_resume["epoch"]

    lr_scheduler = get_scheduler(
        cfg_trainer, len(lab_loader), optimizer, start_epoch=last_epoch
    )
    for epoch in range(last_epoch, cfg['trainer']['epochs']):
        print(epoch, cfg['trainer'].get("sup_only_epoch",1))
        if epoch == cfg['trainer'].get("sup_only_epoch",1):
            model.load_state_dict(torch.load(
                            r'checkpoints/ckpt_best_warm_student_exp_0.0755_1_16.pth')[
                            'model_state'], strict=True)
            model_teacher.load_state_dict(torch.load(
                r'checkpoints/ckpt_best_warm_student_exp_0.0755_1_16.pth')[
                                      'model_state'], strict=True)
        train_one_epoch(
            model,
            model_teacher,
            optimizer,
            lr_scheduler,
            criterion_sup,
            criterion_unsup,
            MSELoss,
            lab_loader,
            unlab_loader,
            epoch,
            write,
            logger,
            test_loader,
            cons_w_unsup
        )
        print('-------start test--------')
        if epoch < cfg["trainer"].get("sup_only_epoch", 1):
            pre_student = validate(model, test_loader)
            pre_teacher = 0
            state = {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "teacher_state": model_teacher.state_dict(),
                    "best_mae": best_student_mae,
                }
            if pre_student < best_student_mae:
                best_student_mae = pre_student
                best_student_epoch = epoch
                torch.save(
                            state, os.path.join(cfg["saver"]["snapshot_dir"], "ckpt_best_warm_student_exp_{:.4f}.pth".format(best_student_mae))
                        )
        else:
            pre_student = validate(model, test_loader)
            pre_teacher = validate(model_teacher, test_loader)
            state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "teacher_state": model_teacher.state_dict(),
                "best_mae": best_student_mae,
            }
            if pre_student < best_student_mae:
                best_student_mae = pre_student
                best_student_epoch = epoch
                torch.save(
                    state, os.path.join(cfg["saver"]["snapshot_dir"],
                                        "ckpt_best_student_exp_{:.4f}.pth".format(best_student_mae))
                )
            if pre_teacher < best_mae:
                best_mae = pre_teacher
                best_epoch = epoch
                torch.save(
                    state, os.path.join(cfg["saver"]["snapshot_dir"],
                                        "ckpt_best_teacher_exp_{:.4f}.pth".format(best_mae))
                )

        torch.save(state, os.path.join(cfg["saver"]["snapshot_dir"], "ckpt.pth"))
        logger.info('#TEST#:Epoch:{} TeacherMAE:{} studentMAE:{} bestEpoch:{} bestMAE:{} bestStudentEpoch:{} bestStudentMAE:{}'.format(epoch, pre_teacher,pre_student,
                                                                                  best_epoch,
                                                                                  best_mae,best_student_epoch, best_student_mae))

def validate(
    model,
    data_loader,
):
    mae_rgb_sum = 0
    model.eval()
    for i in range(data_loader.size):
        image, gt, depth, name, img_for_post = data_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        depth = depth.repeat(1, 3, 1, 1).cuda()
        outs = model(image, depth, name)
        pred, rep = outs["pred"], outs["rep"]

        if cfg['net']['num_classes'] == 2:
            '''取阈值的操作'''
            res_rgb = F.softmax(pred, dim=1)
            res_rgb = torch.max(res_rgb, dim=1)[1].float()
            res_rgb = res_rgb.unsqueeze(1)
            res_rgb = F.interpolate(res_rgb, size=gt.shape, mode='bilinear', align_corners=True)
            res_rgb = res_rgb.squeeze(1)
            res_rgb = res_rgb.data.cpu().numpy().squeeze()

        else:
            res_rgb = F.interpolate(pred, size=gt.shape, mode='bilinear', align_corners=False)
            res_rgb = res_rgb.sigmoid().data.cpu().numpy().squeeze()
        res_rgb = (res_rgb - res_rgb.min()) / (res_rgb.max() - res_rgb.min() + 1e-8)
        mae_rgb_sum += np.sum(np.abs(res_rgb - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])


    mae_rgb = mae_rgb_sum / data_loader.size

    return mae_rgb


import matplotlib.pyplot as plt

def get_feature_RGB(outputs):
    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(
            0)
        gray_scale = torch.sum(feature_map, 0)
        gray_scale = gray_scale / feature_map.shape[
            0]
        processed.append(gray_scale.data.cpu().numpy())
    fig = plt.figure(figsize=(30, 50))

    for i in range(len(processed)):  # len(processed) = 17
        a = fig.add_subplot(5, 4, i + 1)
        img_plot = plt.imshow(processed[i])
    plt.savefig('feature_maps.jpg', bbox_inches='tight')


def train_one_epoch(
    model,
    model_teacher,
    optimizer,
    lr_scheduler,
    sup_loss_fn,
    unsup_loss_fn,
    MSELoss,
    loader_l,
    loader_u,
    epoch,
    tb_logger,
    logger,
    test_loader,
    cons_w_unsup
):
    ema_decay_origin = cfg["net"]["ema_decay"]
    model.train()
    total_step = min(len(loader_l), len(loader_u))

    for step, (data_label, data_unlabel) in enumerate(zip(loader_l, loader_u), 1):
        i_iter = epoch * total_step + step
        image_l, label_l, depth_l = data_label["image"].cuda(), data_label["gt"].cuda(), data_label["depth"].cuda()
        image_l_strong, depth_l_strong = data_label['image_strong'].cuda(), data_label['depth_strong'].cuda()
        name_l = data_label["name"]
        depth_l = depth_l.repeat(1,3,1,1)
        depth_l_strong = depth_l_strong.repeat(1,3,1,1)
        batch_size, _, h, w = label_l.shape
        label_l = label_l.squeeze(1)
        image_u, depth_u, image_u_s = data_unlabel["image_weak"].cuda(), data_unlabel["depth"].cuda(), data_unlabel["image_strong"].cuda()
        depth_u = depth_u.repeat(1, 3, 1, 1)
        #预热阶段
        if epoch < cfg["trainer"].get("sup_only_epoch", 1):
            del image_u, depth_u, image_u_s
            image_l_weak, depth_l_weak, label_l_weak= cut_mix_label_adaptive_ori(
                image_l.clone(),
                depth_l.clone(),
                label_l.clone()
            )
            image_l_strong, depth_l_strong, label_l_mix_strong = cut_mix_label_adaptive_strong(
                image_l_strong.clone(),
                depth_l_strong.clone(),
                label_l.clone(),
            )
            B = image_l.shape[0]
            outs = model(torch.cat((image_l_weak, image_l_strong), dim=0), torch.cat((depth_l_weak, depth_l_strong), dim=0), name_l)
            pred, rep= outs["pred"], outs["rep"]
            pred = F.interpolate(pred, (h,w), mode='bilinear', align_corners=False)
            pred, pred_strong = pred[:B,:, :, :], pred[B:, :, :, :]
            mask1, mask2 =  outs["outr"], outs['outd']
            mask1,mask1_strong = mask1[:B,:, :, :], mask1[B:, :, :, :]
            mask2, mask2_strong = mask2[:B, :, :, :], mask2[B:, :, :, :]
            sup_loss = compute_L_UL_loss(sup_loss_fn,[pred,mask1, mask2, pred_strong, mask1_strong, mask2_strong],
                                         [label_l_weak,label_l_weak, label_l_weak, label_l_mix_strong, label_l_mix_strong, label_l_mix_strong])
            unlabel_weight = 0
            unsup_loss = 0
            all_loss = sup_loss

        else:
            p_threshold = cfg["trainer"]["unsupervised"].get("threshold", 0.95)
            #获取伪标签
            with torch.no_grad():
                model_teacher.eval()
                pred_u_teacher_outs = model_teacher(image_u, depth_u, name_l)
                pred_u_teacher = pred_u_teacher_outs["pred"]
                pred_u_teacher_mask1, pred_u_teacher_mask2 = pred_u_teacher_outs["outr"], pred_u_teacher_outs["outd"]
                label_u_feature = F.softmax(pred_u_teacher, dim=1)
                label_u_teacher_mask1 = F.softmax(pred_u_teacher_mask1, dim=1)
                label_u_teacher_mask2 = F.softmax(pred_u_teacher_mask2, dim=1)
                _, pred_u_teacher_mask1 = torch.max(label_u_teacher_mask1, dim=1)
                _, pred_u_teacher_mask2 = torch.max(label_u_teacher_mask2, dim=1)
                '''获取置信度图'''
                uncertainty_map1 = torch.mean(torch.stack([label_u_feature, label_u_teacher_mask1, label_u_teacher_mask2]), dim=0)
                logits_u, label_u = torch.max(uncertainty_map1, dim=1)
                uncertainty_map1 = -1.0 * torch.sum(uncertainty_map1 * torch.log(uncertainty_map1 + 1e-10), dim=1)
                uncertainty_map1 /= np.log(cfg['net']['num_classes'])
                confindence = 1 - uncertainty_map1
                confindence = confindence * logits_u
                confindence = confindence.mean(dim=[1,2])
                confindence = confindence.cpu().numpy().tolist()

            trigger_prob = cfg["trainer"]["unsupervised"].get("use_cutmix_trigger_prob", 1.0)
            image_l_aug = image_l
            label_l_aug = label_l
            depth_l_aug = depth_l
            del image_l, label_l, depth_l
            if np.random.uniform(0, 1) < trigger_prob and cfg["trainer"]["unsupervised"].get("use_cutmix", False):
                if cfg["trainer"]["unsupervised"].get("use_cutmix_adaptive", False):
                    image_u_aug, depth_u_aug, label_u_aug, logits_u_aug, mask_u_image, mask_u_depth, mask_u_mask, mask_u_logits= cut_mix_label_adaptive(
                        image_u_s.clone(),
                        depth_u.clone(),
                        label_u.clone(),
                        logits_u.clone(),
                        image_u,
                        depth_u,
                        image_l_aug,
                        depth_l_aug,
                        label_l_aug,
                        confindence
                    )

            num_labeled = len(image_l_aug)
            image_all = torch.cat((image_l_aug, image_u_aug))
            depth_all = torch.cat((depth_l_aug, depth_u_aug))
            del image_l_aug,  depth_l_aug
            del image_u_s, depth_u, label_u, logits_u
            del mask_u_image, mask_u_depth

            model.train()
            outs = model(image_all,depth_all, name_l)
            pred_a_all, rep_a_all = outs["pred"], outs["rep"]
            pred_a_all_mask1, pred_a_all_mask2 = outs["outr"], outs["outd"]
            # pred_a_l, pred_a_u, pred_a_u_m = pred_a_all[:num_labeled], pred_a_all[num_labeled:mask_unlabeled], pred_a_all[mask_unlabeled:]
            pred_a_l, pred_a_u= pred_a_all[:num_labeled], pred_a_all[num_labeled:]
            pred_a_l_mask1, pred_a_l_mask2 = pred_a_all_mask1[:num_labeled], pred_a_all_mask2[:num_labeled]
            pred_a_u_mask1, pred_a_u_mask2 = pred_a_all_mask1[num_labeled:], pred_a_all_mask2[num_labeled:]
            del image_u_aug, depth_u_aug
            label_u_aug_mask1 = label_u_aug.clone()
            label_u_aug_mask2 = label_u_aug.clone()
            unsup_loss_pred, _ = compute_unsupervised_loss_by_threshold(pred_a_u, label_u_aug.detach(),
                    logits_u_aug.detach(), thresh=p_threshold)
            unsup_loss_mask1, _ =  compute_unsupervised_loss_by_threshold(pred_a_u_mask1, label_u_aug_mask1.detach(),
                    logits_u_aug.detach(), thresh=p_threshold)
            unsup_loss_mask2, _ = compute_unsupervised_loss_by_threshold(pred_a_u_mask2, label_u_aug_mask2.detach(),
                    logits_u_aug.detach(), thresh=p_threshold)
            unsup_loss = (unsup_loss_pred + unsup_loss_mask1 + unsup_loss_mask2 ) / 3
            del pred_a_u, label_u_aug, logits_u_aug,  label_u_aug_mask1, label_u_aug_mask2
            del mask_u_mask, mask_u_logits
            sup_loss = compute_L_UL_loss(sup_loss_fn, [pred_a_l, pred_a_l_mask1, pred_a_l_mask2],
                                         [label_l_aug, label_l_aug, label_l_aug])
            del pred_a_l, label_l_aug
            unlabel_weight = cons_w_unsup(epoch=epoch, curr_iter=step)
            all_loss = sup_loss + unsup_loss * unlabel_weight

        optimizer.zero_grad()
        all_loss.backward()
        optimizer.step()

        # 更新教师网络
        if epoch >= cfg["trainer"].get("sup_only_epoch", 1):
            with torch.no_grad():
                ema_decay = min( 1 - 1 / ( i_iter - len(loader_l) * cfg["trainer"].get("sup_only_epoch", 1) + 1), ema_decay_origin)
                ##更新weight
                for t_params, s_params in zip(
                        model_teacher.parameters(), model.parameters()
                ):
                    t_params.data = (
                            ema_decay * t_params.data + (1 - ema_decay) * s_params.data
                    )
                ###更新bn
                for t_buffers, s_buffers in zip(
                        model_teacher.buffers(), model.buffers()
                ):
                    t_buffers.data = (
                            ema_decay * t_buffers.data + (1 - ema_decay) * s_buffers.data
                    )


        if step % 20 == 0 or step == total_step or step == 1:
            logger.info(
                '{} #TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR_s:{:.7f}, label_loss:{:4f}, unlabel_loss:{:4f}, unlabel_weight:{:.4f}'.
                format(datetime.now(), epoch, cfg['trainer']['epochs'], step, total_step,
                       optimizer.state_dict()['param_groups'][0]['lr'],
                       sup_loss.data, unsup_loss, unlabel_weight))



if __name__ == '__main__':
    print("Launching...")
    parser = argparse.ArgumentParser(description="Semi-Supervised Semantic Segmentation")
    parser.add_argument("--config", type=str, default="config/sal_config.yaml")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    train(cfg)