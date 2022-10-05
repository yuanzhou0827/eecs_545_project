import signac
import flow
import os
import numpy as np
from tqdm import tqdm
from models import provider
import torch.utils.data
from sklearn.metrics import roc_auc_score
import math
import time

from models.dataset import MasifSiteDataset
import json
# from mpi4py import MPI

pr = signac.get_project()


def make_rng(job):
    """Reproducibly generate an rng from a job."""
    import hashlib
    import random

    random_seed_base = job.sp()
    random_seed = (int(hashlib.sha256(json.dumps(
        random_seed_base, sort_keys=True).encode('utf-8')).hexdigest(), base=16) % (2**32))
    return random.Random(random_seed)


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y,]
    return new_y.cuda()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
        

def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def auc_roc(job, dataset_loader, classifier, n_samples_for_auc=10000, num_classes=1):
    with torch.no_grad():
        classifier = classifier.eval()

        target_sample = []
        pred_sample = []
        n_samples = 0

        for batch_id, (points, _target) in tqdm(
                enumerate(dataset_loader),
                total=len(dataset_loader),
                smoothing=0.9):
            points, target = points.float().cuda(), _target.long().cuda()
            points = points.transpose(2, 1)
            if job.sp.model == 'pointconv':
                seg_pred, _ = classifier(points)
            else:
                seg_pred, _ = classifier(points, to_categorical(0, num_classes))

            seg_pred = seg_pred.contiguous().view(-1, 1).float()[:, 0]
            seg_pred = torch.gt(seg_pred, 0.5).float()
            target = target.contiguous().view(-1, 1).float()[:, 0]
            
            pred_sample.extend(list(seg_pred.cpu().numpy()))
            target_sample.extend(list(target.cpu().numpy()))
        pred_sample = np.array(pred_sample)
        target_sample = np.array(target_sample)

        return roc_auc_score(target_sample, pred_sample)


def get_dataset(job, rng, split='train'):
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    # DATA LOADING
    root = job.fn('')
    dataset = MasifSiteDataset(
        root=root,
        rng=rng,
        dset=job.sp.dset,
        npoint=job.sp.npoint,
        split=split,
        uniform=job.sp.uniform,
        normal_channel=False
    )
    dataset_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=job.sp.batch_size,
        shuffle=True
    )
    return dataset_loader


def load_model(job):
    if job.sp.model == 'pointnet':
        from models.pointnet import seg as model
    elif job.sp.model == 'pointnetpp_msg':
        from models.pointnetpp import seg_msg as model
    elif job.sp.model == 'pointnetpp_ssg':
        from models.pointnetpp import seg_ssg as model
    elif job.sp.model == 'pointconv':
         from models.pointconv import model
    else:
        ferror = ""
        ferror += f"{job.sp.model} is an invalid model type. "
        ferror += "Please select either pointnet, pointnetpp_msg, "
        ferror += "pointnetpp_ssg, or pointconv"
        raise NotImplementedError(ferror)
    return model


class Project(flow.FlowProject):
    pass


@Project.operation.with_directives({'ngpu': 1})
@Project.post(lambda job: job.doc.get('trained') is True)
def train(job):
    rng = make_rng(job)
    train_dataset_loader = get_dataset(job, rng, 'train')
    test_dataset_loader = get_dataset(job, rng, 'test')  # We use test instead of validation due to dataset size
    MODEL = load_model(job)
    classifier = MODEL.get_model(1, normal_channel=False).cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)
    num_part = 2
    num_classes = 1

    if job.isfile('best_model.pth'):
        checkpoint = torch.load(job.fn('best_model.pth'))
        start_epoch = job.doc['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        print('Use pretrain model')
    else:
        print('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)
        with open(job.fn("log.csv"), 'w') as f:
            f.write('epoch,loss,train_auc,test_auc,time\n')
        
    if job.sp.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=job.sp.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=job.sp.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=job.sp.learning_rate, momentum=0.9)

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = job.sp.step_size

    global_epoch = 0
    best_auc = 0

    for epoch in range(start_epoch, job.sp.epoch):
        '''Adjust learning rate and BN momentum'''
        start_time = time.time()
        lr = max(job.sp.learning_rate * (job.sp.lr_decay ** (epoch // job.sp.step_size)), LEARNING_RATE_CLIP)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()

        '''learning one epoch'''
        for i, (points, target) in tqdm(enumerate(train_dataset_loader), total=len(train_dataset_loader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.cpu().data.numpy()
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            if job.sp.model == 'pointconv':
                seg_pred, trans_feat = classifier(points)
            else:
                seg_pred, trans_feat = classifier(points, to_categorical(0, num_classes))
            
            seg_pred = seg_pred.contiguous().view(-1, 1).float()
            target = target.contiguous().view(-1, 1).float()

            if job.sp.model == 'pointconv':
                loss = criterion(seg_pred, target)
            else:
                loss = criterion(seg_pred, target, trans_feat.float())
            loss.backward()
            optimizer.step()

        train_auc = auc_roc(job, train_dataset_loader, classifier)
        print(f"Train AUC-ROC is: {train_auc:.5f}\nLoss: {loss:.5f}")

        test_auc = auc_roc(
            job,
            dataset_loader=test_dataset_loader,
            classifier=classifier
        )

        if test_auc >= best_auc:
            savepath = job.fn('best_model.pth')
            state = {
                'epoch': epoch,
                'train_acc': train_auc,
                'test_auc': test_auc,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)

        diff = time.time() - start_time
        with open(job.fn("log.csv"), 'a') as f:
            f.write(f'{epoch},{loss},{train_auc},{test_auc},{diff}\n')

        if test_auc > best_auc:
            best_auc = test_auc

        job.doc['epoch'] = epoch
        global_epoch += 1
    job.doc['trained'] = True


if __name__ == "__main__":
    Project().main()
