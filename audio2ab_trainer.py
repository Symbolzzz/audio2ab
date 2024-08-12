import train
import os
import time
import csv
import sys
import warnings
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.autograd as autograd

import numpy as np
import time
import pprint
from loguru import logger

from utils import config, logger_tools, other_tools, metric
from dataloaders import data_tools
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func


class CustomTrainer(train.BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.word_rep = args.word_rep
        self.emo_rep = args.emo_rep
        self.sem_rep = args.sem_rep
        self.speaker_id = args.speaker_id
        self.alignmenter = metric.alignment(0.3, 2)
        
        self.loss_meters = {
            'fid_val': other_tools.AverageMeter('fid_val'),
            'rec_val': other_tools.AverageMeter('rec_val'),
            'all': other_tools.AverageMeter('all'),
            'rec': other_tools.AverageMeter('rec'), 
            'gen': other_tools.AverageMeter('gen'),
            'dis': other_tools.AverageMeter('dis'),
            'reg': other_tools.AverageMeter('reg'),
            'kld': other_tools.AverageMeter('kld'),
        } 
    
    def train(self, epoch):
        # 若训练次数达到 no_adv_epochs则使用对抗训练
        use_adv = bool(epoch>=self.no_adv_epochs)
        self.model.train()
        self.d_model.train()
        its_len = len(self.train_loader)
        t_start = time.time()
        for its, batch_data in enumerate(self.train_loader):
#             if its+1 == its_len and tar_pose.shape[0] < self.batchnorm_bug: # skip final bs=1, bug for bn
#                     continue
            t_data = time.time() - t_start
            
            tar_pose = batch_data["pose"].cuda()
            tar_facial = batch_data["facial"].cuda()
            
            in_audio = batch_data["audio"].cuda() if self.audio_rep is not None else None
            # in_facial = batch_data["facial"].cuda() if self.facial_rep is not None else None
            # 不需要speaker id作为训练
            in_id = batch_data["id"].cuda() if self.speaker_id else None
            in_word = batch_data["word"].cuda() if self.word_rep is not None else None
            in_emo = batch_data["emo"].cuda() if self.emo_rep is not None else None
            # 加入情绪标签
            in_sem = batch_data["sem"].cuda() if self.sem_rep is not None else None
            
            tar_combine = torch.cat((tar_pose, tar_facial), dim=2)
            
            # 需要加入 facial feat
            in_pre_pose = tar_pose.new_zeros((tar_pose.shape[0], tar_pose.shape[1], tar_pose.shape[2] + 1)).cuda()
            # 用 4 帧做 seed pose
            in_pre_pose[:, 0:self.pre_frames, :-1] = tar_pose[:, 0:self.pre_frames]
            # 做一个标志位
            in_pre_pose[:, 0:self.pre_frames, -1] = 1 
            
            in_pre_face = tar_facial.new_zeros((tar_facial.shape[0], tar_facial.shape[1], tar_facial.shape[2] + 1)).cuda()
            in_pre_face[:, 0:self.pre_frames, :-1] = tar_facial[:, 0:self.pre_frames]
            in_pre_face[:, 0:self.pre_frames, -1] = 1
        
            t_data = time.time() - t_start
            
            # 使用 WGAN-GP 替换 GAN 
            
            # --------------------------- d training 判别器训练--------------------------------- #
            # 这部分可以先不管，作者说不用对抗生成训练
            d_loss_final = 0
            lambda_gp = 10  # 梯度惩罚的权重
            
            if use_adv:
                # logger.info("Using Adv training...\n")
                self.opt_d.zero_grad()
                # 这里也需要改改？是否使用对抗训练
                # # 禁用 cuDNN
                # with torch.backends.cudnn.flags(enabled=False):
                out_seq = self.model(in_pre_pose, in_pre_face, in_audio=in_audio, in_text=in_word, in_id=in_id, in_emo=in_emo)
                out_pose = out_seq[:, :, :141]
                out_facial = out_seq[:, :, 141:]

                
                out_d_fake = self.d_model(out_seq)
                # d_fake_for_d = self.adv_loss(out_d_fake, fake_gt)
                out_d_real = self.d_model(tar_combine)
                # d_real_for_d = self.adv_loss(out_d_real, real_gt)
                
                # 计算梯度惩罚
                alpha = torch.rand(out_seq.size(0), 1, 1).cuda()
                interpolates_pose = alpha * tar_pose + (1 - alpha) * out_pose
                interpolates_face = alpha * tar_facial + (1 - alpha) * out_facial
                interpolates_face.requires_grad_(True)
                interpolates_face.requires_grad_(True)
                
                interpolates = torch.cat((interpolates_pose, interpolates_face), dim=2)
                out_d_interpolates = self.d_model(interpolates)
                
                gradients_pose = autograd.grad(outputs=out_d_interpolates, inputs=interpolates_pose,
                                        grad_outputs=torch.ones(out_d_interpolates.size()).cuda(),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
                gradients_face = autograd.grad(outputs=out_d_interpolates, inputs=interpolates_face,
                                        grad_outputs=torch.ones(out_d_interpolates.size()).cuda(),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
                
                gradients_pose = gradients_pose.reshape(gradients_pose.size(0), -1)
                gradients_face = gradients_face.reshape(gradients_face.size(0), -1)
                
                gradient_penalty_pose = ((gradients_pose.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
                gradient_penalty_face = ((gradients_face.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
                
                # WGAN-GP判别器损失
                d_loss_adv = torch.mean(out_d_fake) - torch.mean(out_d_real) + gradient_penalty_pose + gradient_penalty_face
                d_loss_final += d_loss_adv
                self.loss_meters['dis'].update(d_loss_final.item()) # we ignore batch_size here
                d_loss_final.backward()
                self.opt_d.step()
                # if lrs_d is not None: lrs_d.step()       
            self.opt.zero_grad()

 
            # --------------------------- g training 生成器训练--------------------------------- #
            g_loss_final = 0
            # 不需要speaker，改成sem
            # 禁用 cuDNN
            # with torch.backends.cudnn.flags(enabled=False):
            out_seq  = self.model(in_pre_pose, in_pre_face, in_audio=in_audio, in_text=in_word, in_id=in_id, in_emo=in_emo)
            out_pose = out_seq[:, :, :141]
            out_facial = out_seq[:, :, 141:]
            
            # 计算重建损失
            if self.sem_rep is not None:
                # 需要改成 combine
                huber_value_pose = self.rec_loss(tar_pose*(in_sem.unsqueeze(2)+1), out_pose*(in_sem.unsqueeze(2)+1))
                huber_value_face = self.rec_loss(tar_facial * (in_sem.unsqueeze(2) + 1), out_facial * (in_sem.unsqueeze(2) + 1))
            else: 
                # 计算两者的 huber loss
                huber_value_pose = self.rec_loss(tar_pose, out_pose)
                huber_value_face = self.rec_loss(tar_facial, out_facial)
            huber_value = huber_value_pose + huber_value_face
            huber_value *= self.rec_weight 
            self.loss_meters['rec'].update(huber_value.item())
            g_loss_final += huber_value 
            if use_adv:
                dis_out = self.d_model(out_seq)
                d_fake_value = -torch.mean(torch.log(dis_out + 1e-8)) # self.adv_loss(out_d_fake, real_gt) # here 1 is real
                d_fake_value *= self.adv_weight 
                self.loss_meters['gen'].update(d_fake_value.item())
                g_loss_final += d_fake_value
                
#                 latent_out = self.eval_model(out_pose)
#                 latent_ori = self.eval_model(tar_pose)
#                 huber_fid_loss = self.rec_loss(latent_out, latent_ori) * self.fid_weight
#                 self.loss_meters[4].update(huber_fid_loss.item())
#                 g_loss_final += huber_fid_loss
            
            self.loss_meters['all'].update(g_loss_final.item())
            g_loss_final.backward()
            if self.grad_norm != 0: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            self.opt.step()
            # if lrs is not None: lrs.step() 
            t_train = time.time() - t_start - t_data
            t_start = time.time()
            mem_cost = torch.cuda.memory_cached() / 1E9
            lr_g = self.opt.param_groups[0]['lr']
            lr_d = self.opt_d.param_groups[0]['lr']
            
            # --------------------------- recording ---------------------------------- #
            if its % self.log_period == 0:
                self.recording(epoch, its, its_len, self.loss_meters, lr_g, lr_d, t_data, t_train, mem_cost)
            #if its == 1:break
        self.opt_s.step(epoch)
        self.opt_d_s.step(epoch)        
    
    
    def val(self, epoch):
        self.model.eval()
        with torch.no_grad():
            its_len = len(self.val_loader)
            for its, batch_data in enumerate(self.val_loader):
#                 if its+1 == its_len and tar_pose.shape[0] < self.batchnorm_bug: # skip final bs=1, bug for bn
#                     continue
                tar_pose = batch_data["pose"].cuda()
                tar_facial = batch_data["facial"].cuda()
                
                in_audio = batch_data["audio"].cuda() if self.audio_rep is not None else None
                in_id = batch_data["id"].cuda() if self.speaker_id else None
                in_word = batch_data["word"].cuda() if self.word_rep is not None else None
                in_emo = batch_data["emo"].cuda() if self.emo_rep is not None else None
                in_sem = batch_data["sem"].cuda() if self.sem_rep is not None else None
                
                # 添加联合特征
                tar_combine = torch.cat((tar_pose, tar_facial), dim=2)
                
                # 添加随机噪声
                noise_pose = torch.randn_like(tar_pose)
                noise_face = torch.rand_like(tar_facial)

                in_pre_pose = tar_pose.new_zeros((tar_pose.shape[0], tar_pose.shape[1], tar_pose.shape[2] + 1)).cuda()
                # in_pre_pose[:, 0:self.pre_frames, :-1] = tar_pose[:, 0:self.pre_frames]
                in_pre_pose[:, 0:self.pre_frames, :-1] = tar_pose[:, 0:self.pre_frames]
                in_pre_pose[:, 0:self.pre_frames, -1] = 1  # indicating bit for constraints
                
                in_pre_face = tar_facial.new_zeros((tar_facial.shape[0], tar_facial.shape[1], tar_facial.shape[2] + 1)).cuda()
                # in_pre_pose[:, 0:self.pre_frames, :-1] = tar_pose[:, 0:self.pre_frames]
                in_pre_face[:, 0:self.pre_frames, :-1] = tar_facial[:, 0:self.pre_frames]
                in_pre_face[:, 0:self.pre_frames, -1] = 1  # indicating bit for constraints
    
                out_seq = self.model(in_pre_pose, in_pre_face, in_audio=in_audio, in_text=in_word, in_id=in_id, in_emo=in_emo)
                out_pose = out_seq[:, :, :141]
                out_facial = out_seq[:, :, 141:]
                # 加入 facial
                latent_out = self.eval_model(out_pose, out_facial)
                latent_ori = self.eval_model(tar_pose, tar_facial)
                
                #print(latent_out,latent_ori)
                if its == 0:
                    latent_out_all = latent_out.cpu().numpy()
                    latent_ori_all = latent_ori.cpu().numpy()
                else:
                    latent_out_all = np.concatenate([latent_out_all, latent_out.cpu().numpy()], axis=0)
                    latent_ori_all = np.concatenate([latent_ori_all, latent_ori.cpu().numpy()], axis=0)
                # 改成 tar_combine
                huber_value_pose = self.rec_loss(tar_pose, out_pose)
                huber_value_face = self.rec_loss(tar_facial, out_facial)
                huber_value = huber_value_pose + huber_value_face
                huber_value *= self.rec_weight
                self.loss_meters['rec_val'].update(huber_value.item())
                #if its == 1:break
            fid = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
            self.loss_meters['fid_val'].update(fid)
            self.val_recording(epoch, self.loss_meters)
                
        
    def test(self, epoch):
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        start_time = time.time()
        total_length = 0
        test_seq_list = os.listdir(self.test_demo)
        # 添加 face 测试
        test_seq_face_list = os.listdir(self.test_demo_face)
        test_seq_list.sort()
        t_start = 10
        t_end = 500
        align = 0 
        self.model.eval()
        with torch.no_grad():
            if not os.path.exists(results_save_path):
                os.makedirs(results_save_path)
            for its, batch_data in enumerate(self.test_loader):
                tar_pose = batch_data["pose"].cuda()
                tar_facial = batch_data["facial"].cuda()
                
                in_audio = batch_data["audio"].cuda() if self.audio_rep is not None else None
                in_id = batch_data["id"].cuda() if self.speaker_id else None
                in_word = batch_data["word"].cuda() if self.word_rep is not None else None
                in_emo = batch_data["emo"].cuda() if self.emo_rep is not None else None
                in_sem = batch_data["sem"].cuda() if self.sem_rep is not None else None
                
                # 添加联合特征
                tar_combine = torch.cat((tar_pose, tar_facial), dim=2)
                
                pre_pose = tar_pose.new_zeros((tar_pose.shape[0], tar_pose.shape[1], tar_pose.shape[2] + 1)).cuda()
                # pre_pose[:, 0:self.pre_frames, :-1] = tar_pose[:, 0:self.pre_frames]
                pre_pose[:, 0:self.pre_frames, :-1] = tar_pose[:, 0:self.pre_frames]
                pre_pose[:, 0:self.pre_frames, -1] = 1
                
                pre_face = tar_facial.new_zeros((tar_facial.shape[0], tar_facial.shape[1], tar_facial.shape[2] + 1)).cuda()
                # in_pre_pose[:, 0:self.pre_frames, :-1] = tar_pose[:, 0:self.pre_frames]
                pre_face[:, 0:self.pre_frames, :-1] = tar_facial[:, 0:self.pre_frames]
                pre_face[:, 0:self.pre_frames, -1] = 1  # indicating bit for constraints
                
                in_audio = in_audio.reshape(1, -1)   
                out_dir_vec = self.model(**dict(pre_seq=pre_pose, pre_seq_facial=pre_face, in_audio=in_audio, in_text=in_word, in_id=in_id, in_emo=in_emo))
                # 分别处理 pose 和 facial
                out_final = out_dir_vec.cpu().numpy().reshape(-1, self.pose_dims + self.facial_dims)
                out_final_pose = (out_final[:, :141] * self.std_pose) + self.mean_pose
                out_final_face = (out_final[:, 141:] * self.std_face) + self.mean_face
                #out_final = out_dir_vec.cpu().numpy().reshape(-1, self.pose_dims) + self.mean_pose
                total_length += out_final_pose.shape[0]
                #print(out_final.shape)
                
                onset_raw, onset_bt, onset_bt_rms = self.alignmenter.load_audio(in_audio.cpu().numpy().reshape(-1), t_start, t_end, True)
                beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist = self.alignmenter.load_pose(out_final_pose, t_start, t_end, self.pose_fps, True)
                align += self.alignmenter.calculate_align(onset_raw, onset_bt, onset_bt_rms, beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist, self.pose_fps)
                #print(align)
                res_pose = "pose/"
                res_face = "facial52/"
                
                if not os.path.exists(f"{results_save_path}{res_pose}"):
                    os.mkdir(f"{results_save_path}{res_pose}")
                with open(f"{results_save_path}{res_pose}result_raw_{test_seq_list[its]}", 'w+') as f_real:
                    for line_id in range(out_final_pose.shape[0]): #,args.pre_frames, args.pose_length
                        line_data = np.array2string(out_final_pose[line_id], max_line_width=np.inf, precision=6, suppress_small=False, separator=' ')
                        f_real.write(line_data[1:-2]+'\n')  
                        
                        
                if not os.path.exists(f"{results_save_path}{res_face}"):
                    os.mkdir(f"{results_save_path}{res_face}") 
                with open(f"{results_save_path}{res_face}result_raw_{test_seq_face_list[its]}", 'w+') as f_face_real:
                    for line_id in range(out_final_face.shape[0]):
                        line_data = np.array2string(out_final_face[line_id], max_line_width=np.inf, precision=6, suppress_small=False, separator=' ')
                        f_face_real.write(line_data[1:-2] + '\n')
                        
                        
        align_avg = align/len(self.test_loader)
        logger.info(f"align score: {align_avg}")
        data_tools.result2target_vis(self.pose_version, results_save_path + res_pose, results_save_path, self.test_demo, False)
        data_tools.resulit2target_facial(result_jsonlist=results_save_path + res_face, save_path=results_save_path + res_face)
        
        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.pose_fps)} s motion")
               
    @staticmethod
    def diversity(output, clips):
        pass
    
    @staticmethod
    def SRGR(output, target, weight, alpha=0.2):
        pass