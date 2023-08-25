def recall(feature_vectors, feature_labels, rank,test_img_list, gallery_vectors=None, gallery_labels=None):
    num_features = len(feature_labels)
    feature_labels = torch.tensor(feature_labels, device=feature_vectors.device)
    gallery_vectors = feature_vectors if gallery_vectors is None else gallery_vectors

    dist_matrix = torch.cdist(feature_vectors.unsqueeze(0), gallery_vectors.unsqueeze(0)).squeeze(0)
    
    if gallery_labels is None:
        dist_matrix.fill_diagonal_(float('inf'))
        gallery_labels = feature_labels
    else:
        gallery_labels = torch.tensor(gallery_labels, device=feature_vectors.device)

    idx = dist_matrix.topk(k=rank[-1], dim=-1, largest=False)[1]
    acc_list = []
    for r in rank:
        correct = (gallery_labels[idx[:, 0:r]] == feature_labels.unsqueeze(dim=-1)).any(dim=-1).float()
        acc_list.append((torch.sum(correct) / num_features).item())
    return acc_list
def test(net, recall_ids):
    net.eval()
    with torch.no_grad():
        # obtain feature vectors for all data
        for key in eval_dict.keys():
            eval_dict[key]['features'] = []
            imgList=[]
            # for inputs, labels in tqdm(eval_dict[key]['data_loader'], desc='processing {} data'.format(key)):
            print('processing {} data'.format(key))
            for batch_idx, (inputs, labels, imgs) in enumerate(eval_dict[key]['data_loader']):
                inputs, labels = inputs.cuda(), labels.cuda()
                features, classes = net(inputs)
                eval_dict[key]['features'].append(features)
                imgList+=imgs
            eval_dict[key]['features'] = torch.cat(eval_dict[key]['features'], dim=0)

        # compute recall metric
        if data_name == 'isc':
            acc_list = recall(eval_dict['test']['features'], test_data_set.label_list, recall_ids,imgList,
                              eval_dict['gallery']['features'], gallery_data_set.label_list)
        else:
            acc_list = recall(eval_dict['test']['features'], test_data_set.label_list, recall_ids,imgList)
    
    desc = 'Test Epoch {}/{} '.format(epoch, num_epochs)
    for index, rank_id in enumerate(recall_ids):
        desc += 'R@{}:{:.2f}% '.format(rank_id, acc_list[index] * 100)
        results['test_recall@{}'.format(rank_id)].append(acc_list[index] * 100)
    print(desc)
    return acc_list[0]

import argparse

import pandas as pd
import torch
from thop import profile, clever_format
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import SingleData
from model import Model, set_bn_eval
from utils import  LabelSmoothingCrossEntropyLoss, BatchHardTripletLoss, ImageReader, MPerClassSampler
import numpy as np
import os
import math 
import torchvision.transforms as transforms

def get_data_list(data_path, ratio=0.001):
    img_list = []
    for root, dirs, files in os.walk(data_path):
        if files == []:
            class_name = dirs
        elif dirs == []:
            for f in files:
                img_path = os.path.join(root, f)
                img_list.append(img_path)

    np.random.seed(1)
    train_img_list = np.random.choice(img_list, size=int(len(img_list)*(1-ratio)), replace=False)
    #print(img_list, train_img_list)
    eval_img_list = list(set(img_list) - set(train_img_list))
    ########add
    half=math.floor(len(eval_img_list)/2)
    print(half)
    eval_=eval_img_list[:half]
    test_=eval_img_list[half:]
    #######
    #return class_name, train_img_list, eval_img_list 
    return class_name, train_img_list


def train(net, optim):
    net.train()
    # fix bn on backbone network
    net.apply(set_bn_eval)
    # total_loss, total_correct, total_num, data_bar = 0, 0, 0, tqdm(train_data_loader)
    print("start")
    total_loss, total_correct, total_num, data_bar = 0, 0, 0, enumerate(train_data_loader)
    print("end")
    # for inputs, labels in data_bar:
    for batch_idx, (inputs, labels, _) in data_bar:
        print(batch_idx)
        inputs, labels = inputs.cuda(), labels.cuda()
        features, classes = net(inputs)
        class_loss = class_criterion(classes, labels)
        feature_loss = feature_criterion(features, labels)
        loss = class_loss + feature_loss
        optim.zero_grad()
        loss.backward()
        optim.step()
        pred = torch.argmax(classes, dim=-1)
        total_loss += loss.item() * inputs.size(0)
        total_correct += torch.sum(pred == labels).item()
        total_num += inputs.size(0)
        print('Train Epoch {}/{} - Loss:{:.4f} - Acc:{:.2f}%'.format(epoch, num_epochs, total_loss / total_num, total_correct / total_num * 100))
        # data_bar.set_description('Train Epoch {}/{} - Loss:{:.4f} - Acc:{:.2f}%'
        #                          .format(epoch, num_epochs, total_loss / total_num, total_correct / total_num * 100))

    return total_loss / total_num, total_correct / total_num * 100




if __name__ == '__main__':
    
    
    # parser = argparse.ArgumentParser(description='Train CGD')
    # parser.add_argument('--data_path', default='/home/data', type=str, help='datasets path')
    # parser.add_argument('--data_name', default='car', type=str, choices=['car', 'cub', 'sop', 'isc'],
    #                     help='dataset name')
    # parser.add_argument('--crop_type', default='uncropped', type=str, choices=['uncropped', 'cropped'],
    #                     help='crop data or not, it only works for car or cub dataset')
    # parser.add_argument('--backbone_type', default='resnet50', type=str, choices=['resnet50', 'resnext50'],
    #                     help='backbone network type')
    # parser.add_argument('--gd_config', default='SG', type=str,
    #                     choices=['S', 'M', 'G', 'SM', 'MS', 'SG', 'GS', 'MG', 'GM', 'SMG', 'MSG', 'GSM'],
    #                     help='global descriptors config')
    # parser.add_argument('--feature_dim', default=1536, type=int, help='feature dim')
    # parser.add_argument('--smoothing', default=0.1, type=float, help='smoothing value for label smoothing')
    # parser.add_argument('--temperature', default=0.5, type=float,
    #                     help='temperature scaling used in softmax cross-entropy loss')
    # parser.add_argument('--margin', default=0.1, type=float, help='margin of m for triplet loss')
    # parser.add_argument('--recalls', default='1,2,4,8', type=str, help='selected recall')
    # parser.add_argument('--batch_size', default=128, type=int, help='train batch size')
    # parser.add_argument('--num_epochs', default=20, type=int, help='train epoch number')

    # opt = parser.parse_args()
    # args parse
    data_path="C:/Users/user01/Documents/datasets/new_bracs/new"
    # data_path="D:\\models\\CGD-master\\dataset\\img"
    test_path="E:/dataset/test/test"
    data_name="CRC"
    crop_type="uncropped"
    backbone_type="resnet50"
    gd_config="SM"
    feature_dim=1536
    smoothing=0.1
    temperature=0.5
    margin=0.1
    tempRecall='1,2,4,8'
    recalls=[int(k) for k in tempRecall.split(',')]
    batch_size=8
    num_epochs=29
    
    class_name, train_img_list = get_data_list(data_path)
    class_test_name, test_img_list = get_data_list(test_path)
    
    train_transform = transforms.Compose([ 
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomCrop(224),
    transforms.ToTensor()
])
    
    
    # data_path, data_name, crop_type, backbone_type = opt.data_path, opt.data_name, opt.crop_type, opt.backbone_type
    # gd_config, feature_dim, smoothing, temperature = opt.gd_config, opt.feature_dim, opt.smoothing, opt.temperature
    # margin, recalls, batch_size = opt.margin, [int(k) for k in opt.recalls.split(',')], opt.batch_size
    # num_epochs = opt.num_epochs
    save_name_pre = '{}_{}_{}_{}_{}_{}_{}_{}_{}_gs_pretrian'.format(data_name, crop_type, backbone_type, gd_config, feature_dim,
                                                        smoothing, temperature, margin, batch_size)

    results = {'train_loss': [], 'train_accuracy': []}
#     resultsWithC = {'train_loss': [], 'train_accuracy': []}
    
    for recall_id in recalls:
        results['test_recall@{}'.format(recall_id)] = []
#         resultsWithC['test_recall@{}'.format(recall_id)] = []
        

    train_data_set=SingleData(class_name, train_img_list, train_transform)
    # dataset loader
    # train_data_set = ImageReader(data_path, data_name, 'train', crop_type)
    # train_sample = MPerClassSampler(train_data_set.labels, batch_size)
    train_sample = MPerClassSampler(train_data_set.label_list, batch_size)
    train_data_loader = DataLoader(train_data_set, batch_sampler=train_sample, num_workers=8)
    # test_data_set = ImageReader(data_path, data_name, 'query' if data_name == 'isc' else 'test', crop_type)
    test_data_set = SingleData(class_test_name, test_img_list, train_transform)
    test_data_loader = DataLoader(test_data_set, batch_size, shuffle=False, num_workers=8)
    eval_dict = {'test': {'data_loader': test_data_loader}}
    if data_name == 'isc':
        gallery_data_set = ImageReader(data_path, data_name, 'gallery', crop_type)
        gallery_data_loader = DataLoader(gallery_data_set, batch_size, shuffle=False, num_workers=8)
        eval_dict['gallery'] = {'data_loader': gallery_data_loader}

    # model setup, model profile, optimizer config and loss definition
    model = Model(backbone_type, gd_config, feature_dim, num_classes=3).cuda()
    # model.load_state_dict(torch.load('D:\\models\\CGD-master_2\\CGD-master\\isc_uncropped_resnet50_GS_1536_0.1_0.5_0.1_128_model.pth'))

    flops, params = profile(model, inputs=(torch.randn(1, 3, 224, 224).cuda(),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    optimizer = Adam(model.parameters(), lr=1e-4)
    lr_scheduler = MultiStepLR(optimizer, milestones=[int(0.6 * num_epochs), int(0.8 * num_epochs)], gamma=0.1)
    class_criterion = LabelSmoothingCrossEntropyLoss(smoothing=smoothing, temperature=temperature)
    feature_criterion = BatchHardTripletLoss(margin=margin)

    best_recall = 0.0
    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = train(model, optimizer)
        results['train_loss'].append(train_loss)
        results['train_accuracy'].append(train_accuracy)
#         resultsWithC['train_loss'].append(train_loss)
#         resultsWithC['train_accuracy'].append(train_accuracy)
        rank = test(model, recalls)
#         rank2 = testWithTask(model, recalls)
        
        lr_scheduler.step()

        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
#         data_frame_with_condition = pd.DataFrame(data=resultsWithC, index=range(1, epoch + 1))
        
        data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
#         data_frame_with_condition.to_csv('results/{}_statistics_with_condition.csv'.format(save_name_pre), index_label='epoch')
        
        # save database and model
        data_base = {}
        if rank > best_recall:
            best_recall = rank
            data_base['test_images'] = test_data_set.img_list
            data_base['test_labels'] = test_data_set.label_list
            data_base['test_features'] = eval_dict['test']['features']
            if data_name == 'isc':
                data_base['gallery_images'] = gallery_data_set.img_list
                data_base['gallery_labels'] = gallery_data_set.label_list
                data_base['gallery_features'] = eval_dict['gallery']['features']
            torch.save(model.state_dict(), 'results/{}_{}_model.pth'.format(save_name_pre,epoch))
            torch.save(data_base, 'results/{}_{}_data_base.pth'.format(save_name_pre,epoch))
