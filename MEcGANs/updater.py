import numpy as np

import chainer
import chainer.functions as F
from chainer import Variable

import cv2
import cupy as cp

# Classic Adversarial Loss
def loss_dcgan_dis(dis_fake, dis_real):
    L1 = F.mean(F.softplus(-dis_real))
    L2 = F.mean(F.softplus(dis_fake))
    loss = L1 + L2
    return loss


def loss_dcgan_gen(dis_fake):
    loss = F.mean(F.softplus(-dis_fake))
    return loss


# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real):
    loss = F.mean(F.relu(1. - dis_real))
    loss += F.mean(F.relu(1. + dis_fake))
    return loss


def loss_hinge_gen(dis_fake):
    loss = -F.mean(dis_fake)
    return loss


class LossL1:
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, x, t):
        if self.weight == 0:
            return Variable(np.array(0.))
        else:
            return F.mean_absolute_error(x, t) * self.weight


class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.models = kwargs.pop('models')
        self.n_dis = kwargs.pop('n_dis')
        self.loss_type = kwargs.pop('loss_type')
        if self.loss_type == 'dcgan':
            self.loss_gen = loss_dcgan_gen
            self.loss_dis = loss_dcgan_dis
        elif self.loss_type == 'hinge':
            self.loss_gen = loss_hinge_gen
            self.loss_dis = loss_hinge_dis
        else:
            raise NotImplementedError
        self.loss_l1 = LossL1(weight=kwargs.pop('weight_l1'))
        super(Updater, self).__init__(*args, **kwargs)

    def get_batch(self, xp):
        batch = self.get_iterator('main').next()
        batchsize = len(batch)

        x = []
        y = []
        for j in range(batchsize):
            x.append(np.asarray(batch[j][0]).astype("f"))
            y.append(np.asarray(batch[j][1]).astype("f"))
        x = Variable(xp.asarray(x))
        y = Variable(xp.asarray(y))

        return x, y
        
    def get_rgbcloudedges_batch(self, xp, rgbcloud_batch):
        batchsize = len(rgbcloud_batch)
        
        #rgbcloudedges_batch = []
        rgbcloudedges_batch = xp.ndarray([batchsize,5,256,256],dtype=np.dtype(np.float32))
        for j in range(batchsize):
            # Get an rgbcloud image (generated or ground-truth) from the batch.
            # It is an (4,256,256) array.
            #img = np.asarray(rgbcloud_batch[j]).astype("f")
            img = (rgbcloud_batch[j]).array
            # Get the RGB part and make it into an (256,256,3) array with values in
            # [0, 255].
            img_rgb = xp.clip(img.transpose(1,2,0)[:,:,:3] * 127.5 + 127.5,0.,255.).astype(np.uint8)
            # Compute the edge-filtered image. The threshold_max value should be
            # adjusted.
            
            # This is slow!!!
            # Move from GPU to CPU
            img_rgb_np = cp.asnumpy(img_rgb)
            edges = cv2.Canny(img_rgb_np,0,100,apertureSize=3)
            # Move back to GPU
            edges = cp.asarray(edges)
            
            # Make into an (256,256) array with values in [-1,1].
            edges = edges / 127.5 - 1.
            #rgbcloudedges = xp.concatenate((img,edges[None,:,:]),axis=0)
            #rgbcloudedges_batch.append(rgbcloudedges)
            rgbcloudedges_batch[j,:,:,:] = xp.concatenate((img,edges[None,:,:]),axis=0)

        #rgbcloudedges_batch = Variable(xp.asarray(rgbcloudedges_batch))
        rgbcloudedges_batch = Variable(rgbcloudedges_batch)

        return rgbcloudedges_batch

    def update_core(self):
        gen = self.models['gen']
        dis = self.models['dis']
        gen_optimizer = self.get_optimizer('opt_gen')
        dis_optimizer = self.get_optimizer('opt_dis')
        xp = gen.xp
        x, y = self.get_batch(xp)
        
        # Based on data.py NIRRGB2RGBCLOUD class
        # First dimension is the batch dimension.
        # x - nirrgb; x[:,0,:,:] - clouded NIR, x[:,1:4,:,:] - clouded RGB
        # y - rgbcloud; y[:,:3,:,:] - target RGB (ground truth), y[:,3,:,:] - cloud
        # Based on evaluation.py out_image() and save_images()
        # y_fake = gen(x); y_fake[:,:3,:,:] - generated cloud-free RGB,
        #                  y_fake[:,3,:,:]  - cloud mask
        
        y_edges = self.get_rgbcloudedges_batch(xp, y)
        
        for i in range(self.n_dis):
            if i == 0:
                y_fake = gen(x)
                
                y_fake_edges = self.get_rgbcloudedges_batch(xp, y_fake)
                
                dis_fake = dis(x, y_fake_edges)
                loss_gen = self.loss_gen(dis_fake=dis_fake)
                loss_l1 = self.loss_l1(y_fake, y)
                gen.cleargrads()
                loss_gen.backward()
                loss_l1.backward()
                gen_optimizer.update()
                chainer.reporter.report({'loss_gen': loss_gen})
                chainer.reporter.report({'loss_l1': loss_l1})

            x, y = self.get_batch(xp)
            
            y_edges = self.get_rgbcloudedges_batch(xp, y)
            
            dis_real = dis(x, y_edges)
            y_fake = gen(x)
            
            y_fake_edges = self.get_rgbcloudedges_batch(xp, y_fake)
            
            dis_fake = dis(x, y_fake_edges)
            y_fake_edges.unchain_backward()

            loss_dis = self.loss_dis(dis_fake=dis_fake, dis_real=dis_real)
            dis.cleargrads()
            loss_dis.backward()
            dis_optimizer.update()
            chainer.reporter.report({'loss_dis': loss_dis})
