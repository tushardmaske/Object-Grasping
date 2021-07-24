import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_sched
from pointnet2_ops.pointnet2_modules import PointnetSAModule
import numpy as np


def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(lr_sched.LambdaLR):
    def __init__(self, model, bn_lambda, last_epoch=-1, setter=set_bn_momentum_default):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(type(model)._name_)
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def state_dict(self):
        return dict(last_epoch=self.last_epoch)

    def load_state_dict(self, state):
        self.last_epoch = state["last_epoch"]
        self.step(self.last_epoch)

lr_clip = 1e-5
bnm_clip = 1e-2

class PointNetEncoderDecoder(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self._build_model()
        self.batch_size=hparams["batch_size"]

    def _build_model(self):
        self.encoder = nn.ModuleList()
        self.encoder.append(
            PointnetSAModule(
                npoint=self.hparams["num_points"],
                radius=0.2,
                nsample=64,
                mlp=[0, 64, 64, 128],
                use_xyz=True,
            )
        )
        self.encoder.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=64,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.encoder.append(
            PointnetSAModule(
                mlp=[256, 256, 512, 1024], use_xyz=True
            )
        )

        self.decoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
        )

        self.euler_orn = nn.Linear(256, 3)
        self.euler_pos = nn.Linear(256, 3)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features


    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.encoder:
            xyz, features = module(xyz, features)

        z = self.decoder(features.squeeze(-1))
        act = torch.cat((self.euler_pos(z),self.euler_orn(z)), -1)

        return act

    def training_step(self, batch, batch_idx):
        pc, labels = batch
        pc = pc.float()
        labels = labels.float()

        logits = self.forward(pc)
        loss=self.Point_matching_loss(logits,labels)
        log = dict(loss=loss)
        return dict(loss=loss, log=log)

    def validation_step(self, batch, batch_idx):
        pc, labels = batch
        pc=pc.float()
        labels=labels.float()

        logits = self.forward(pc)
        val_loss = self.Point_matching_loss(logits, labels)
        log = dict(val_loss=val_loss)
        return dict(val_loss=val_loss,log=log)


    def configure_optimizers(self):
        lr_lbmd = lambda _: max(
            self.hparams["optimizer.lr_decay"]
            ** (
                int(
                    self.global_step
                    * self.hparams["batch_size"]
                    / self.hparams["optimizer.decay_step"]
                )
            ),
            lr_clip / self.hparams["optimizer.lr"],
        )
        bn_lbmd = lambda _: max(
            self.hparams["optimizer.bn_momentum"]
            * self.hparams["optimizer.bnm_decay"]
            ** (
                int(
                    self.global_step
                    * self.hparams["batch_size"]
                    / self.hparams["optimizer.decay_step"]
                )
            ),
            bnm_clip,
        )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams["optimizer.lr"],
            weight_decay=self.hparams["optimizer.weight_decay"],
        )
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd)
        bnm_scheduler = BNMomentumScheduler(self, bn_lambda=bn_lbmd)

        return [optimizer], [lr_scheduler, bnm_scheduler]

    def Point_matching_loss (self,logits,labels):

        expert_action_batch = labels.cuda()
        grasp_pc = get_control_point_tensor(len(labels), device='cuda', rotz=True)
        gt_act_pt = control_points_from_rot_and_trans(expert_action_batch[:, 3:6], expert_action_batch[:, :3],
                                                      device='cuda', grasp_pc=grasp_pc)
        # pi, _, _, grasp_pred = self.policy.sample(state_batch)
        action_trans_mean = torch.abs(logits[..., :3]).mean()

        # bc_loss = self.bc_loss(gt_act_pt[expert_mask], pi[expert_mask], grasp_pc[expert_mask],
        #                        expert_action_batch[expert_mask])

        pred_act_pt = control_points_from_rot_and_trans(logits[: self.batch_size, 3:6],logits[: self.batch_size, :3],
                    device="cuda", grasp_pc=grasp_pc)

        bc_loss = torch.mean(torch.abs(pred_act_pt - gt_act_pt).sum(-1))
        return bc_loss

    def GraspLoss(self,pred,GT):

        ss=nn.CrossEntropyLoss()
        GT[GT == -1] = 0
        GT=GT.long()
        gloss=ss(pred,GT)
        return gloss

    # def optimize(self, loss, update_step):
    #     """
    #     Backward loss and update optimizer
    #     """
    #     self.state_feat_encoder_optim.zero_grad()
    #     self.policy_optim.zero_grad()
    #     loss.backward()  #

def get_control_point_tensor(batch_size, use_torch=True, device="cpu", rotz=False):
    """
    Outputs a tensor of shape (batch_size x 6 x 3).
    use_tf: switches between outputing a tensor and outputing a numpy array.
    """
    control_points = np.array([[0., 0., 0.],
                               [0., 0., 0.],
                               [0.053, -0., 0.075],
                               [-0.053, 0., 0.075],
                               [0.053, -0., 0.105],
                               [-0.053, 0., 0.105]], dtype=np.float32)
    control_points = np.tile(np.expand_dims(control_points, 0), [batch_size, 1, 1])
    if rotz:
        control_points = np.matmul(control_points, rotZ(np.pi / 2)[:3, :3])
    if use_torch:
        return torch.tensor(control_points).to(device).float()

    return control_points.astype(np.float32)

def rotZ(rotz):
    RotZ = np.array(
        [
            [np.cos(rotz), -np.sin(rotz), 0, 0],
            [np.sin(rotz), np.cos(rotz), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    return RotZ

def control_points_from_rot_and_trans(
    grasp_eulers, grasp_translations, device="cpu", grasp_pc=None
):
    rot = tc_rotation_matrix(
        grasp_eulers[:, 0], grasp_eulers[:, 1], grasp_eulers[:, 2], batched=True
    )
    if grasp_pc is None:
        grasp_pc = get_control_point_tensor(grasp_eulers.shape[0], device=device)

    grasp_pc = torch.matmul(grasp_pc.float(), rot.permute(0, 2, 1))
    grasp_pc += grasp_translations.unsqueeze(1).expand(-1, grasp_pc.shape[1], -1)
    return grasp_pc


def tc_rotation_matrix(az, el, th, batched=False):
    if batched:

        cx = torch.cos(torch.reshape(az, [-1, 1]))
        cy = torch.cos(torch.reshape(el, [-1, 1]))
        cz = torch.cos(torch.reshape(th, [-1, 1]))
        sx = torch.sin(torch.reshape(az, [-1, 1]))
        sy = torch.sin(torch.reshape(el, [-1, 1]))
        sz = torch.sin(torch.reshape(th, [-1, 1]))

        ones = torch.ones_like(cx)
        zeros = torch.zeros_like(cx)

        rx = torch.cat([ones, zeros, zeros, zeros, cx, -sx, zeros, sx, cx], dim=-1)
        ry = torch.cat([cy, zeros, sy, zeros, ones, zeros, -sy, zeros, cy], dim=-1)
        rz = torch.cat([cz, -sz, zeros, sz, cz, zeros, zeros, zeros, ones], dim=-1)

        rx = torch.reshape(rx, [-1, 3, 3])
        ry = torch.reshape(ry, [-1, 3, 3])
        rz = torch.reshape(rz, [-1, 3, 3])

        return torch.matmul(rz, torch.matmul(ry, rx))
    else:
        cx = torch.cos(az)
        cy = torch.cos(el)
        cz = torch.cos(th)
        sx = torch.sin(az)
        sy = torch.sin(el)
        sz = torch.sin(th)

        rx = torch.stack([[1.0, 0.0, 0.0], [0, cx, -sx], [0, sx, cx]], dim=0)
        ry = torch.stack([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dim=0)
        rz = torch.stack([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dim=0)

        return torch.matmul(rz, torch.matmul(ry, rx))
