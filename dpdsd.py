#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Runs MNIST training with differential privacy.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np
import random

from model import SampleConvNet
from load_data import load_data_fashionmnist
from parse_args import fashionmnist_parse_args

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    set_random_seed(42)

    args = fashionmnist_parse_args()
    device = torch.device(args.device)

    epsilon = 3.0

    train_loader, test_loader = load_data_fashionmnist(args, args.batch_size)
    model = SampleConvNet().to(device)
    teacher_model = SampleConvNet().to(device)

    # 此处用到了 lr_z
    if args.optim == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        teacher_optimizer = optim.SGD(
            teacher_model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        raise NotImplementedError(
            "Optimizer not recognized. Please check spelling")

    privacy_engine = None
    if not args.disable_dp:
        privacy_engine = PrivacyEngine(
            secure_mode=args.secure_rng,
        )
        clipping = "per_layer" if args.clip_per_layer else "flat"

        # 此处用到了norm_z and epsilon_z, epochs_z
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=args.epochs,
            target_epsilon=epsilon,
            max_grad_norm=args.max_per_sample_grad_norm,
            clipping=clipping,
            grad_sample_mode=args.grad_sample_mode,
            target_delta=1e-5
        )

        teacher_model, teacher_optimizer, teacher_train_loader = privacy_engine.make_private_with_epsilon(
            module=teacher_model,
            optimizer=teacher_optimizer,
            data_loader=train_loader,
            epochs=args.epochs,
            target_epsilon=epsilon,
            max_grad_norm=args.max_per_sample_grad_norm,
            clipping=clipping,
            grad_sample_mode=args.grad_sample_mode,
            target_delta=1e-5
        )

    # 使用StepLR调度器，每10个epoch将学习率乘以0.9
    scheduler_step = StepLR(optimizer, step_size=10, gamma=0.8)
    best_acc1 = 0

    for epoch in range(0, args.epochs):

        if epoch > 0:
            teacher_model.load_state_dict(model.state_dict())
            # train dpdsd
            epsilon_epoch = train_dpdsd(
                args, model, optimizer, teacher_model,
                train_loader,
                privacy_engine, epoch, device,
                args.confidence,
                args.alpha, args.beta, args.temperature,
            )
        else:
            # train dpsgd
            epsilon_epoch = train_dpsgd(
                args, model,
                optimizer,
                train_loader,
                privacy_engine,
                epoch, device)

        scheduler_step.step()
        top1_acc, test_loss = test(args, model, test_loader, device,
                                   True)
        best_acc1 = max(top1_acc, best_acc1)

    print(f"best_accuracy: {best_acc1}")


def train_dpdsd(args, model, optimizer, teacher_model,
               train_loader,
               privacy_engine, epoch, device,
               confidence_threshold=0.1,
               alpha=1.0, beta=0.1, temperature=2
               ):

    model.train()
    criterion = nn.CrossEntropyLoss(reduction='none')  # hard label loss, 无缩减，逐个计算损失

    for i, (images, target) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)

        with torch.no_grad():
            teacher_output = teacher_model(images)
            teacher_output_kd = teacher_output.detach()

        loss_ce = criterion(output, target)

        ndsd_loss_value = ndsd_loss3(output, teacher_output_kd, target,
                                     alpha, beta,
                                     confidence_threshold, temperature
                                     )

        loss = loss_ce + ndsd_loss_value

        loss = loss.mean()  # 计算批次平均损失

        # compute gradient and do SGD step
        loss.backward()

        # make sure we take a step after processing the last mini-batch in the
        # epoch to ensure we start the next epoch with a clean state
        optimizer.step()
        optimizer.zero_grad()

    # epsilon_epoch = privacy_engine.accountant.get_epsilon(delta=args.delta)
    epsilon_epoch = privacy_engine.get_epsilon(delta=args.delta)

    return epsilon_epoch


def train_dpsgd(args, model, optimizer,
                train_loader,
                privacy_engine, epoch, device):

    model.train()
    criterion = nn.CrossEntropyLoss()

    for i, (images, target) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)

        loss = criterion(output, target)

        # compute gradient and do SGD step
        loss.backward()

        # make sure we take a step after processing the last mini-batch in the
        # epoch to ensure we start the next epoch with a clean state
        optimizer.step()
        optimizer.zero_grad()

    if not args.disable_dp:
        epsilon_epoch = privacy_engine.accountant.get_epsilon(delta=args.delta)
    else:
        epsilon_epoch = 0

    return epsilon_epoch


def ndsd_loss3(student_logits, teacher_logits,  target,
               alpha=1.0, beta=0.1, confidence=0.1,
               temperature=2.0):
    """
    DKD loss function with dynamic beta based on teacher confidence.
    :param teacher_logits: logits from the teacher model
    :param student_logits: logits from the student model
    :param target: ground truth labels
    :param alpha: weight for TCKD
    :param base_beta: base value for beta, will be adjusted based on confidence
    :param temperature: temperature for distillation
    """
    T = temperature

    # Compute probabilities with stability
    epsilon = 1e-10  # small constant to prevent log(0)
    teacher_probs = F.softmax(teacher_logits / T, dim=1).clamp(min=epsilon, max=1.0)
    student_probs = F.softmax(student_logits / T, dim=1).clamp(min=epsilon, max=1.0)

    # Get target mask and compute pt and pnt
    target_mask = F.one_hot(target, num_classes=10).bool()
    pt_teacher = teacher_probs[target_mask].view(-1, 1).clamp(min=epsilon, max=1.0)
    pt_student = student_probs[target_mask].view(-1, 1).clamp(min=epsilon, max=1.0)

    # Construct complete distributions for target and non-target classes
    full_teacher_probs = torch.cat([pt_teacher, 1 - pt_teacher], dim=1).clamp(min=epsilon, max=1.0)
    full_student_probs = torch.cat([pt_student, 1 - pt_student], dim=1).clamp(min=epsilon, max=1.0)

    # Calculate TCKD using full distributions
    TCKD = F.kl_div(full_student_probs.log(), full_teacher_probs, reduction='none') * (T ** 2)
    TCKD_per_sample = TCKD.sum(dim=1)

    pnt_teacher = teacher_probs.masked_fill(target_mask, 0).clamp(min=epsilon, max=1.0)
    pnt_student = student_probs.masked_fill(target_mask, 0).clamp(min=epsilon, max=1.0)

    pnt_teacher_sum = pnt_teacher.sum(dim=1, keepdim=True).clamp(min=epsilon, max=1.0)
    pnt_student_sum = pnt_student.sum(dim=1, keepdim=True).clamp(min=epsilon, max=1.0)

    # Calculate NCKD for each sample with reduction='none'
    pnt_student_normalized = (pnt_student / pnt_student_sum).clamp(min=epsilon, max=1.0)
    pnt_teacher_normalized = (pnt_teacher / pnt_teacher_sum).clamp(min=epsilon, max=1.0)
    NCKD = F.kl_div(pnt_student_normalized.log(), pnt_teacher_normalized, reduction='none') * (T ** 2)

    # Sum over classes to get per-sample loss
    NCKD_per_sample = NCKD.sum(dim=1)

    # Adjust beta based on confidence for each sample
    confidence_diff = (pt_teacher - confidence)

    # Combine losses
    loss = (alpha * TCKD_per_sample + beta * NCKD_per_sample) * torch.exp(confidence_diff)
    return loss


def set_random_seed(seed_value):
    torch.manual_seed(seed_value)  # 设置 PyTorch 随机种子
    torch.cuda.manual_seed_all(seed_value)  # 如果使用多个 GPU，也应该添加这行
    np.random.seed(seed_value)  # 设置 NumPy 的随机种子
    random.seed(seed_value)  # 设置 Python 原生随机库的种子
    torch.backends.cudnn.deterministic = True  # 确保 CNN 的一致性
    torch.backends.cudnn.benchmark = False  # 可提高训练速度，但在不同运行中可能会导致微小差异


def test(args, model, test_loader, device, flag=True):
    """
    flag=True 表明是测试数据集，Fasle 代表的是训练数据集
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    correct = 0

    with torch.no_grad():
        for images, target in tqdm(test_loader):
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)

            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            losses.append(loss.item())

    top1_avg2 = correct / len(test_loader.dataset)

    return top1_avg2, losses

if __name__ == "__main__":
    main()
