# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np

def linear_rampup(args, current, rampup_length=50):
    current = np.clip(current / rampup_length, 0.0, 1.0)
    return args.lambda_u * float(current)

class attention_mix_net(nn.Module):
    def __init__(self, net, n_classes=10):
        super(attention_mix_net, self).__init__()
        self.net = net
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(net.feature_dim, n_classes)

        # Init the fc layer
        nn.init.kaiming_normal_(self.fc.weight.data)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias.data, val=0)

    def forward(self, input_data):
        feature_maps = self.net.get_features(input_data)
        feature_vector = self.avgpool(feature_maps).view(feature_maps.size(0), -1)
        logits = self.fc(feature_vector)

        return logits, feature_maps

def cutmix(inputs, targets):
    W = inputs.size(2)
    H = inputs.size(3)
    lam = np.random.beta(1, 1)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    idx = torch.randperm(inputs.size(0))
    # input_a, input_b = all_inputs, all_inputs[idx]
    target_a, target_b = targets, targets[idx]
    inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[idx, :, bbx1:bbx2, bby1:bby2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
    targets = lam*target_a + (1-lam)*target_b

    return inputs, targets, 0


def mixup(inputs, targets):
    idx = torch.randperm(inputs.size(0))
    input_a, input_b = inputs, inputs[idx]
    target_a, target_b = targets, targets[idx]

    l = np.random.beta(1, 1)
    l = max(l, 1 - l)
    mixed_input = l * input_a + (1 - l) * input_b
    mixed_target = l * target_a + (1 - l) * target_b

    return mixed_input, mixed_target, 0


def attention_cutmix(inputs, attentions, targets, threshold):
    attentions_numpy = attentions.cpu().numpy()
    threshold = np.percentile(attentions_numpy, 43/49*100, axis=(2, 3), keepdims=True)#(N, 1, 1, 1)
    threshold = torch.tensor(threshold).cuda()
    mask = attentions >= threshold  # (N, 1, H, W)
    mixed_region = torch.mean(torch.sum(mask, dim=[1, 2, 3], keepdim=False) / (attentions.size(2) * attentions.size(3)))
    weight = torch.sum(mask, dim=(1, 2, 3), keepdim=True) / (attentions.size(2)*attentions.size(3))

    idx = torch.randperm(inputs.size(0))
    inputs_shuffle = inputs[idx]  # (3N, C, H, W)
    targets_shuffle = targets[idx]

    mixed_input = inputs * mask + inputs_shuffle * (~mask)
    mixed_target = targets * weight[:, :, 0, 0] + targets_shuffle * (1 - weight[:, :, 0, 0])
    #mixed_target = mixed_target / torch.sum(mixed_target, dim=1, keepdim=True)
    #print(weight[0], targets[0], targets_shuffle[0], mixed_target[0])

    return mixed_input, mixed_target, mixed_region


def attention_mix(inputs, attentions, targets, threshold):
    mask = attentions >= threshold #(3N, 1, H, W)
    mix_region = torch.mean(torch.sum(mask, dim=[1, 2, 3], keepdim=False) / (attentions.size(2) * attentions.size(3)))

    activate = torch.sum(attentions * mask, dim=(1, 2, 3), keepdim=True)
    all = torch.sum(attentions, dim=(1, 2, 3), keepdim=True) #(3N, 1, 1, 1)
    W = activate / (all + 1e-12) #(b + 1e-12) #(3N, 1, 1, 1)

    idx = torch.randperm(inputs.size(0))
    inputs_shuffle = inputs[idx] #(3N, C, H, W)
    attentions_shuffle = attentions[idx] #(3N, 1, H, W)
    targets_shuffle = targets[idx]

    W_corelation = torch.sum(torch.mul(mask, attentions_shuffle), dim=(1, 2, 3), keepdim=True) / (torch.sum(attentions_shuffle, dim=(1, 2, 3), keepdim=True) + 1e-12) #(3N, 1, 1, 1)

    lambda_1 = 1 - W
    lambda_2 = W / (W + W_corelation + 1e-12) * W
    lambda_3 = W - lambda_2  # W_corelation / (W + W_corelation) * w

    mixed_inputs = inputs * (~mask) + W / (W + W_corelation + 1e-12) * (mask * inputs) + W_corelation / (W + W_corelation + 1e-12) * (mask * inputs_shuffle)
    mixed_labels = (lambda_1 + lambda_2)[:, :, 0, 0] * targets + lambda_3[:, :, 0, 0] * targets_shuffle
    #mixed_labels = mixed_labels / torch.sum(mixed_labels, dim=1, keepdim=True)

    return mixed_inputs, mixed_labels, mix_region


def makeCAM(feature_maps, net, target):
    fc_weight = net.fc.weight.data #(200, 2048)
    target = torch.argmax(target, dim=1, keepdim=False) #(3N)
    weight = fc_weight[target] #(3N, 2048)
    weight = weight.view(weight.size(0), weight.size(1), 1, 1) #(3N, 2048, 1, 1)
    cam = torch.sum(weight * feature_maps, dim=1, keepdim=True) #(3N, 1, 7, 7)
    attention = F.interpolate(cam, size=(32, 32)) #(3N, 1, 224, 224)

    _max = torch.max(torch.max(attention, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]  # (3N, 1, 1, 1)
    _min = torch.min(torch.min(attention, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]  # (3N, 1, 1, 1)
    attention = (attention - _min) / (_max - _min)

    return attention #(3N, 1, 224, 224)

def feature_attention(feature_maps, size):
    attention = torch.sum(torch.pow(feature_maps, 2), dim=1, keepdim=True) #(N, 1, 7, 7)
    attention = F.interpolate(attention, size=(size, size)) #(N, 1, 32, 32)

    _max = torch.max(torch.max(attention, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]  # (N, 1, 1, 1)
    _min = torch.min(torch.min(attention, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]  # (N, 1, 1, 1)
    attention = (attention - _min) / (_max - _min + 1e-12)

    return attention #(N, 1, 224, 224)

def train_attention_mix(args, epoch, net, optimizer, labeled_trainloader):
    net.train()

    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1
    mixed_region_sum = 0
    for batch_idx, (inputs_x, inputs_x2, _, labels_x, poisoned) in enumerate(labeled_trainloader):
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1, 1), 1)
        poisoned = poisoned.view(-1, 1).type(torch.FloatTensor)
        inputs_x, inputs_x2, labels_x, poisoned = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), poisoned.cuda()


        with torch.no_grad():
            _, feature_maps1 = net(inputs_x) #(2N, attention_num, H, W)
            _, feature_maps2 = net(inputs_x2)  # (2N, attention_num, H, W)

            all_inputs_x = torch.cat([inputs_x, inputs_x2], dim=0)  # (3N, 3, H, W)
            feature_maps = torch.cat([feature_maps1, feature_maps2], dim=0)
            all_labels = torch.cat([labels_x, labels_x], dim=0) #(3N, 200)

            #all_attentions = makeCAM(feature_maps, net, all_labels)
            all_attentions = feature_attention(feature_maps, inputs_x.size(2))
            all_attentions = all_attentions.detach()

            mixed_inputs, mixed_labels, mixed_region = attention_mix(all_inputs_x, all_attentions, all_labels, threshold=0.1)
            #mixed_inputs, mixed_labels, mixed_region = attention_cutmix(all_inputs_x, all_attentions, all_labels, threshold=0.1)
            #mixed_inputs, mixed_labels, mixed_region = mixup(all_inputs_x, all_labels)
            #mixed_inputs, mixed_labels, mixed_region = cutmix(all_inputs_x, all_labels)
            mixed_region_sum += mixed_region

        logits1, _ = net(mixed_inputs[:batch_size, ...])
        logits2, _ = net(mixed_inputs[batch_size:, ...])
        logits = torch.cat([logits1, logits2], dim=0)

        loss = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_labels, dim=1))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        sys.stdout.write('\r')
        sys.stdout.write('%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f  Mixed region: %.2f'
                         % (args.dataset, epoch, args.num_epochs, batch_idx + 1, num_iter,
                            loss.item(), 0, mixed_region_sum/(batch_idx+1)))
        sys.stdout.flush()


def train_attention_mix_mixmatch(args, epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader, prob2, pred2, loader):
    net.train()
    net2.eval()

    labeled_train_iter = iter(labeled_trainloader)
    num_iter = len(labeled_trainloader.dataset) // args.batch_size + 1
    batch_size_u = (len(pred2) - len(labeled_trainloader.dataset)) // num_iter
    _, unlabeled_trainloader = loader.run('train_net1', pred2, prob2, batch_size_u)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    count = 0
    mixed_region_sum = 0
    correct = 0
    sum = 0
    wrong = 0
    sum1 = 0
    sum_u = 0
    correct_u = 0
    wrong_u = 0
    sum1_u = 0
    for batch_idx in range(0, num_iter):
        try:
            inputs_x, inputs_x2, inputs_x3, labels_x, poisoned = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, inputs_x2, inputs_x3, labels_x, poisoned = labeled_train_iter.next()
        try:
            inputs_u, inputs_u2, inputs_u3, labels_u, poisoned_u = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, inputs_u3, labels_u, poisoned_u = unlabeled_train_iter.next()

        batch_size = inputs_x.size(0)
        batch_size_u = inputs_u.size(0)

        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1, 1), 1)
        labels_u = torch.zeros(batch_size_u, args.num_class).scatter_(1, labels_u.view(-1, 1), 1)

        poisoned = poisoned.view(-1, 1).type(torch.FloatTensor)
        poisoned_u = poisoned_u.view(-1, 1).type(torch.FloatTensor)

        inputs_x, inputs_x2, inputs_x3, labels_x, poisoned = inputs_x.cuda(), inputs_x2.cuda(), inputs_x3.cuda(), labels_x.cuda(), poisoned.cuda()
        inputs_u, inputs_u2, inputs_u3, labels_u, poisoned_u = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda(), labels_u.cuda(), poisoned_u.cuda()

        with torch.no_grad():
            outputs_u11, feature_maps_u = net(inputs_u)
            outputs_u12, _ = net(inputs_u2)
            outputs_u_net2 = net2(inputs_u3)
            outputs_x_net2 = net2(inputs_x3)

            targets_u_net2 = torch.softmax(outputs_u_net2/3, dim=1)
            targets_u_net2 = torch.cat([targets_u_net2, targets_u_net2], dim=0).detach()

            targets_x_net2 = torch.softmax(outputs_x_net2/3, dim=1)
            targets_x_net2 = torch.cat([targets_x_net2, targets_x_net2], dim=0).detach()


            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1)) / 2
            mask = (pu == pu.max(dim=1, keepdim=True)[0]).to(dtype=torch.int32)
            ptu = torch.mul(mask, pu)
            targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
            targets_u = targets_u.detach()

            _, feature_maps1 = net(inputs_x)  # (2N, attention_num, H, W)
            _, feature_maps2 = net(inputs_x2)  # (2N, attention_num, H, W)

            all_inputs_x = torch.cat([inputs_x, inputs_x2], dim=0)  # (3N, 3, H, W)
            feature_maps = torch.cat([feature_maps1, feature_maps2], dim=0)
            all_labels = torch.clip(torch.cat([labels_x, labels_x], dim=0) - targets_x_net2, 0, 1)

            all_attentions = feature_attention(feature_maps, inputs_x.size(2))
            all_attentions = all_attentions.detach()

            mixed_inputs, mixed_labels, mixed_region = attention_mix(all_inputs_x, all_attentions, all_labels, threshold=0.1)
            mixed_region_sum += mixed_region


        logits_mix1, _ = net(mixed_inputs[:batch_size, ...])
        logits_mix2, _ = net(mixed_inputs[batch_size:2*batch_size, ...])

        logits_u1, _ = net(inputs_u)
        logits_u2, _ = net(inputs_u2)

        logits_mix = torch.cat([logits_mix1, logits_mix2], dim=0)
        logits_u = torch.cat([logits_u1, logits_u2], dim=0)

        targets_mix = mixed_labels
        targets_u = torch.cat([targets_u, targets_u], dim=0)

        poisoned = torch.cat([poisoned, poisoned], dim=0)
        poisoned_u = torch.cat([poisoned_u, poisoned_u], dim=0)
        labels_x = torch.cat([labels_x, labels_x], dim=0)
        labels_u = torch.cat([labels_u, labels_u], dim=0)
        for i in range(2*batch_size):
            pred = torch.argmax(targets_x_net2[i])
            if poisoned[i]:
                sum += 1
                if pred == args.trigger_label:
                    correct += 1
            else:
                sum1 += 1
                pred1 = torch.argmax(labels_x[i])
                if pred == pred1:
                    wrong += 1
        for i in range(2*batch_size_u):
            pred_u = torch.argmax(targets_u[i])
            if poisoned_u[i]:
                sum_u += 1
                if pred_u == args.trigger_label:
                    wrong_u += 1
            else:
                sum1_u += 1
                pred1 = torch.argmax(labels_u[i])
                if pred1 == pred_u:
                    correct_u += 1

        Lmix = -torch.mean(torch.sum(F.log_softmax(logits_mix, dim=1) * targets_mix, dim=1))
        Lu = -torch.mean(torch.sum(F.log_softmax(logits_u, dim=1) * torch.clip(targets_u - targets_u_net2, 0, 1), dim=1))

        if epoch <= 150:
            lam = args.alpha1
        else:
            lam = args.alpha2
        loss = Lmix + lam*Lu

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        sys.stdout.write('\r')
        sys.stdout.write(
            '%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Mixed loss: %.2f  Unlabeled loss: %.2f  Mixed region: %.2f |Clean set: Correct: %d/%d  Wrong: %d/%d '
            '|Poison set: Correct: %d/%d  Wrong: %d/%d'
            % (args.dataset, epoch, args.num_epochs, batch_idx + 1, num_iter,
               Lmix.item(), Lu.item(), mixed_region_sum / (batch_idx + 1), correct, sum, wrong, sum1, correct_u, sum1_u, wrong_u, sum_u))
        sys.stdout.flush()
    print('\nDelete: ' + str(count))
