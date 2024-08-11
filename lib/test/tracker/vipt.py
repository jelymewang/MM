import random

import math
from torch import optim
from torch.autograd import Variable

from lib.models.vipt import build_viptrack
from lib.test.tracker.basetracker import BaseTracker
import torch
import numpy as np
import torch.nn.functional as F
from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
import vot
from lib.test.tracker.data_utils import PreprocessorMM
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
import torchvision.utils as tvu
from memory_profiler import profile
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage


class ViPTTrack(BaseTracker):
    def __init__(self, params):
        super(ViPTTrack, self).__init__(params)
        network = build_viptrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = PreprocessorMM()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        if getattr(params, 'debug', None) is None:
            setattr(params, 'debug', 0)
        self.use_visdom = False #params.debug
        self.debug = params.debug
        self.frame_id = 0
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr  = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr)
        with torch.no_grad():
            self.z_tensor = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.device, template_bbox)

        # save state
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}
    def track_d(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1

        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)

        with torch.no_grad():
            x_tensor = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_tensor, search=x_tensor, ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        orig_response = self.output_window * pred_score_map
        if self.frame_id==1:
            last_per1 = 0
            iteration = 10
        elif self.frame_id % 30 == 0:
            last_per1 = 0
            iteration = 10
        else:
            last_per1 = self.last_per1
            iteration = 5
        epsilon = 5
        alpha = epsilon/iteration
        x_patch_arr = torch.tensor(x_patch_arr).cuda().float().permute((2, 0, 1)).unsqueeze(dim=0)
        x_crop = x_patch_arr[:, 3:, :, :]
        t_crop = x_patch_arr[:, :3, :, :]
        x_crop_init = x_crop + last_per1
        x_crop_init = torch.clamp(x_crop_init, 0, 255)
        criterion = torch.nn.KLDivLoss(reduction='batchmean')
        x_adv = x_crop_init.clone().detach()
        x_adv.requires_grad = True
        for i in range(iteration):
            if i==0:
                delta = torch.zeros_like(x_crop)
                delta.data.uniform_(0.0, 1.0)
                delta.data = delta.data * epsilon
                search = self.preprocessor.process_adv(torch.cat((t_crop,x_adv+delta),dim=1))
            else:
                search =  self.preprocessor.process_adv(torch.cat((t_crop,x_adv),dim=1))
            x_tensor = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_tensor, search=x_tensor, ce_template_mask=self.box_mask_z)

            # add hann windows
            pred_score_map = out_dict['score_map']
            adv_response = self.output_window * pred_score_map
            # loss_embed = criterion(adv_embed.log_softmax(dim=-1), orig_embed.softmax(dim=-1))
            loss_heatmap = criterion(adv_response.log_softmax(dim=-1), orig_response.softmax(dim=-1))
            loss = loss_heatmap
            self.network.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)
            loss.backward()
            x_adv.retain_grad()
            grad = x_adv.grad
            sign_grad = grad.sign()
            x_adv = x_adv + sign_grad * alpha
            x_adv = x_crop + torch.clip(x_adv-x_crop,-5,5)
            x_adv = torch.clip(x_adv,0,255)
            x_adv = x_adv.clone().detach()
            x_adv.requires_grad = True
        self.last_per1 = x_adv - x_crop
        search =  self.preprocessor.process_adv(torch.cat((t_crop,x_adv),dim=1))
        with torch.no_grad():
            x_tensor = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_tensor, search=x_tensor, ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes, best_score = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'], return_score=True)
        max_score = best_score[0][0].item()
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # for debug
        if self.debug == 1:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image[:,:,:3], cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
            cv2.putText(image_BGR, 'max_score:' + str(round(max_score, 3)), (40, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 255), 2)
            cv2.imshow('debug_vis', image_BGR)
            cv2.waitKey(1)

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save,
                    "best_score": max_score}
        else:
            return {"target_bbox": self.state,
                    "best_score": max_score}
    def track_t(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)

        with torch.no_grad():
            x_tensor = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_tensor, search=x_tensor, ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        orig_response = self.output_window * pred_score_map
        if self.frame_id==1:
            last_per1 = 0
            iteration = 10
        elif self.frame_id % 30 == 0:
            last_per1 = 0
            iteration = 10
        else:
            last_per1 = self.last_per1
            iteration = 5
        epsilon = 5
        alpha = epsilon/iteration
        x_patch_arr = torch.tensor(x_patch_arr).cuda().float().permute((2, 0, 1)).unsqueeze(dim=0)
        x_crop = x_patch_arr[:, 3:4, :, :]
        t_crop = x_patch_arr[:, 0:3, :, :]
        x_crop_init = x_crop + last_per1
        x_crop_init = torch.clamp(x_crop_init, 0, 255)
        criterion = torch.nn.KLDivLoss(reduction='batchmean')
        x_adv = x_crop_init.clone().detach()
        x_adv.requires_grad = True
        for i in range(iteration):
            if i==0:
                delta = torch.zeros_like(x_crop)
                delta.data.uniform_(0.0, 1.0)
                delta.data = delta.data * epsilon
                search = self.preprocessor.process_adv(
                    torch.cat((t_crop, torch.cat([x_adv + delta] * 3, dim=1)), dim=1))
            else:
                search = self.preprocessor.process_adv(
                    torch.cat((t_crop, torch.cat([x_adv] * 3, dim=1)), dim=1))
            x_tensor = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_tensor, search=x_tensor, ce_template_mask=self.box_mask_z)

            # add hann windows
            pred_score_map = out_dict['score_map']
            adv_response = self.output_window * pred_score_map
            # loss_embed = criterion(adv_embed.log_softmax(dim=-1), orig_embed.softmax(dim=-1))
            loss_heatmap = criterion(adv_response.log_softmax(dim=-1), orig_response.softmax(dim=-1))
            loss = loss_heatmap
            self.network.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)
            loss.backward()
            x_adv.retain_grad()
            grad = x_adv.grad
            sign_grad = grad.sign()
            x_adv = x_adv + sign_grad * alpha
            x_adv = x_crop + torch.clip(x_adv-x_crop,-5,5)
            x_adv = torch.clip(x_adv,0,255)
            x_adv = x_adv.clone().detach()
            x_adv.requires_grad = True
        self.last_per1 = x_adv - x_crop
        search = self.preprocessor.process_adv(
            torch.cat((t_crop, torch.cat([x_adv] * 3, dim=1)), dim=1))
        # search = self.preprocessor.process_adv(x_adv)
        with torch.no_grad():
            x_tensor = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_tensor, search=x_tensor, ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes, best_score = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'], return_score=True)
        max_score = best_score[0][0].item()
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # for debug
        if self.debug == 1:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image[:,:,:3], cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
            cv2.putText(image_BGR, 'max_score:' + str(round(max_score, 3)), (40, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 255), 2)
            cv2.imshow('debug_vis', image_BGR)
            cv2.waitKey(1)

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save,
                    "best_score": max_score}
        else:
            return {"target_bbox": self.state,
                    "best_score": max_score}

    def track_e(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)

        with torch.no_grad():
            x_tensor = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_tensor, search=x_tensor, ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        orig_response = self.output_window * pred_score_map
        iteration = 1
        x_patch_arr = torch.tensor(x_patch_arr).cuda().float().permute((2, 0, 1)).unsqueeze(dim=0)
        x_crop = x_patch_arr[:, 3:, :, :]
        t_crop = x_patch_arr[:, 0:3, :, :]
        criterion = torch.nn.KLDivLoss(reduction='batchmean')
        x_adv = x_crop.clone().detach()
        x_adv.requires_grad = True
        for i in range(iteration):

            search =  self.preprocessor.process_adv(torch.cat((t_crop, x_adv), dim=1))
            x_tensor = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_tensor, search=x_tensor, ce_template_mask=self.box_mask_z)

            # add hann windows
            pred_score_map = out_dict['score_map']
            adv_response = self.output_window * pred_score_map
            # loss_embed = criterion(adv_embed.log_softmax(dim=-1), orig_embed.softmax(dim=-1))
            loss_heatmap = criterion(adv_response.log_softmax(dim=-1), orig_response.softmax(dim=-1))
            loss = loss_heatmap
            self.network.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)
            loss.backward()
            x_adv.retain_grad()
            grad = x_adv.grad

            def random_color_points(original_img, grad):

                _, _, height, width = original_img.shape
                original_tmp = original_img.squeeze(dim=0).permute((1, 2, 0)).clone()
                grad = grad.squeeze(dim=0).permute((1, 2, 0))

                black = torch.tensor([0,0,0], dtype=torch.int).cuda()
                white = torch.tensor([255,255,255], dtype=torch.int).cuda()
                red = torch.tensor([220, 30, 30], dtype=torch.int).cuda()
                blue = torch.tensor([30, 30, 200], dtype=torch.int).cuda()
                mask = ~torch.all(original_tmp == black, dim=-1)
                non_zero_indices = torch.nonzero(mask)
                count_red = torch.sum(torch.all(original_tmp == red, axis=-1))
                count_blue = torch.sum(torch.all(original_tmp == blue, axis=-1))
                #
                num_points = math.ceil((count_blue + count_red) / 2)

                if num_points == 0:
                    return original_img

                grad_masked = grad[mask]
                grad_sum = torch.sum(grad_masked, dim=-1)
                sorted_indices = torch.argsort(torch.abs(grad_sum), descending=True)
                top_points_index = sorted_indices[:num_points]
                grad_sum_selected = grad_sum[top_points_index]
                grad_selected = grad_masked[top_points_index]
                mask1 = (grad_sum_selected < 0) & (grad_selected[:, 0] > grad_selected[:, 2])
                mask2 = (grad_sum_selected < 0) & (grad_selected[:, 2] > grad_selected[:, 0])
                mask3 = grad_sum_selected > 0
                replace1 = torch.tensor([220., 30., 30.], device='cuda').unsqueeze(0)
                replace2 = torch.tensor([30., 30., 200.], device='cuda').unsqueeze(0)
                replace3 = torch.tensor([255., 255., 255.], device='cuda').unsqueeze(0)
                index1 = non_zero_indices[top_points_index[mask1]]
                index2 = non_zero_indices[top_points_index[mask2]]
                index3 = non_zero_indices[top_points_index[mask3]]
                original_tmp[index1[:, 0], index1[:, 1]] = torch.cat([replace1] * index1.size(0), dim=0) if index1.size(
                    0) != 0 else original_tmp[index1[:, 0], index1[:, 1]]
                original_tmp[index2[:, 0], index2[:, 1]] = torch.cat([replace2] * index2.size(0), dim=0) if index2.size(
                    0) != 0 else original_tmp[index2[:, 0], index2[:, 1]]
                original_tmp[index3[:, 0], index3[:, 1]] = torch.cat([replace3] * index3.size(0), dim=0) if index3.size(
                    0) != 0 else original_tmp[index3[:, 0], index3[:, 1]]

                image_noise = original_tmp.float().permute((2, 0, 1)).unsqueeze(dim=0)

                return image_noise

            with torch.no_grad():
                x_adv = random_color_points(x_adv, grad)
            x_adv = x_adv.clone().detach()
            x_adv.requires_grad = True
        search =  self.preprocessor.process_adv(torch.cat((t_crop, x_adv), dim=1))
        with torch.no_grad():
            x_tensor = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_tensor, search=x_tensor, ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes, best_score = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'], return_score=True)
        max_score = best_score[0][0].item()
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        # for debug
        if self.debug == 1:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image[:,:,:3], cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
            cv2.putText(image_BGR, 'max_score:' + str(round(max_score, 3)), (40, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 255), 2)
            cv2.imshow('debug_vis', image_BGR)
            cv2.waitKey(1)

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save,
                    "best_score": max_score}
        else:
            return {"target_bbox": self.state,
                    "best_score": max_score}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)
        with torch.no_grad():
            x_tensor = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_tensor, search=x_tensor, ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes, best_score = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'], return_score=True)
        max_score = best_score[0][0].item()
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # for debug
        if self.debug == 1:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image[:,:,:3], cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
            cv2.putText(image_BGR, 'max_score:' + str(round(max_score, 3)), (40, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 255), 2)
            cv2.imshow('debug_vis', image_BGR)
            cv2.waitKey(1)

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save,
                    "best_score": max_score}
        else:
            return {"target_bbox": self.state,
                    "best_score": max_score}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return ViPTTrack
