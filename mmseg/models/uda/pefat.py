import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from copy import deepcopy
from matplotlib import pyplot as plt

from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import denorm, strong_transform, get_mean_std
from mmseg.models.utils.visualization import subplotimg
from mmseg.models.losses.contrastive_loss import SpatialInfoNCELoss, SpatialKLDivLoss

try:
    from sklearn.mixture import GaussianMixture
except ImportError:
    GaussianMixture = None

def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    return torch.norm(torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

@UDA.register_module()
class PEFAT(UDADecorator):
    def __init__(self, **cfg):
        super(PEFAT, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.warmup_iters = cfg.get('warmup_iters', 10000)
        self.alpha = cfg['alpha']
        self.K_steps = cfg.get('pefat_k_steps', 1) # default 1 for memory reasons
        
        self.color_jitter_s = cfg.get('color_jitter_strength', 0.2)
        self.color_jitter_p = cfg.get('color_jitter_probability', 0.2)
        self.blur = cfg.get('blur', True)
        self.debug_img_interval = cfg.get('debug_img_interval', 1000)

        ema_cfg = deepcopy(cfg['model'])
        self.ema_model = build_segmentor(ema_cfg)

        self.contrastive_loss = SpatialInfoNCELoss(temperature=0.07)
        self.kl_loss = SpatialKLDivLoss()

        self.loss_buffer = []  # For GMM fitting
        self.ema_gmm = None
        
        # We need a cross entropy loss with no reduction for GMM
        self.pixel_ce = nn.CrossEntropyLoss(reduction='none')

    def get_ema_model(self):
        return get_module(self.ema_model)

    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:
                ema_param.data = alpha_teacher * ema_param.data + (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]

    def extract_features(self, model, img):
        # Extract features and logits
        # For memory, we limit what we extract
        feat = model.extract_feat(img)
        out = model._decode_head_forward_test(feat, model.img_metas if hasattr(model, "img_metas") else None)
        # resize out
        out = F.interpolate(out, size=img.shape[2:], mode='bilinear', align_corners=False)
        return feat, out
    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def forward_train(self, img, img_metas, gt_semantic_seg, target_img, target_img_metas):
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        if self.local_iter == 0:
            self._init_ema_weights()

        if self.local_iter > 0:
            self._update_ema(self.local_iter)

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': np.random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': np.random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),
            'std': stds[0].unsqueeze(0)
        }

        # Source CE supervised training
        clean_losses = self.get_model().forward_train(img, img_metas, gt_semantic_seg, return_feat=True)
        src_feat_list = clean_losses.pop('features')
        src_feat = src_feat_list[-1] # Usually take the last representation layer
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        clean_loss.backward(retain_graph=True)

        ###########################################
        # STAGE 1: Warmup & InfoNCE
        ###########################################
        if self.local_iter < self.warmup_iters:
            # Generate two augmentations for InfoNCE (Weak vs Strong on target)
            weak_target_img = target_img
            strong_target_img, _ = strong_transform(
                strong_parameters,
                data=target_img.clone(),
                target=torch.zeros_like(gt_semantic_seg)
            )

            # Extract features for contrastive
            weak_feat = self.get_model().extract_feat(weak_target_img)[-1]
            strong_feat = self.get_model().extract_feat(strong_target_img)[-1]
            
            los_nce = self.contrastive_loss(weak_feat, strong_feat)
            log_vars['loss_pefat_nce'] = los_nce.item()
            los_nce.backward()

            # Collect pseudo-labels for GMM fitting
            with torch.no_grad():
                ema_logits = self.get_ema_model().encode_decode(target_img, target_img_metas)
                pseudo_prob, pseudo_label = torch.max(torch.softmax(ema_logits, dim=1), dim=1)
                
                # We calculate CE on the current batch's weak prediction vs pseudo label to see uncertainty
                student_logits = self.get_model().encode_decode(weak_target_img, target_img_metas)
                loss_map = self.pixel_ce(student_logits, pseudo_label) # [B, H, W]
                
                # Start collecting at end of warmup to save memory
                if self.local_iter > self.warmup_iters - 100:
                    subsample = loss_map.view(-1)[torch.randperm(loss_map.numel())[:100]].cpu().numpy()
                    self.loss_buffer.extend(subsample)
        
        ###########################################
        # GMM Fitting Trigger
        ###########################################
        elif self.local_iter == self.warmup_iters:
            if GaussianMixture is None:
                raise ImportError("Please install scikit-learn for GMM fitting: pip install scikit-learn")
            
            print(f"\n[PEFAT] Fitting GMM over {len(self.loss_buffer)} samples...")
            all_losses = np.array(self.loss_buffer).reshape(-1, 1)
            # Normalize losses 0-1
            pmin, pmax = all_losses.min(), all_losses.max()
            if pmax > pmin:
                all_losses = (all_losses - pmin) / (pmax - pmin)
            
            self.ema_gmm = GaussianMixture(n_components=2, max_iter=50, tol=1e-2, reg_covar=5e-4)
            self.ema_gmm.fit(all_losses)
            self._gmm_pmin = pmin
            self._gmm_pmax = pmax
            
            # Predict the means to know which is trustworthy (lower loss)
            self._trust_idx = self.ema_gmm.means_.argmin()
            print(f"[PEFAT] GMM Fitting Complete. Trust mean: {self.ema_gmm.means_[self._trust_idx][0]}")

        ###########################################
        # STAGE 2: Adversarial Training
        ###########################################
        if self.local_iter > self.warmup_iters and self.ema_gmm is not None:
            # 1) Generate EMA Pseudo labels
            with torch.no_grad():
                ema_logits = self.get_ema_model().encode_decode(target_img, target_img_metas)
                ema_smax = torch.softmax(ema_logits, dim=1)
                _, pseudo_label = torch.max(ema_smax, dim=1)
                
                # Forward target image to get loss map
                t_logits = self.get_model().encode_decode(target_img, target_img_metas)
                loss_map = self.pixel_ce(t_logits, pseudo_label) # [B, H, W]
                
                # Normalize loss map for GMM
                norm_loss_map = (loss_map - self._gmm_pmin) / (self._gmm_pmax - self._gmm_pmin + 1e-8)
                norm_loss_map = norm_loss_map.cpu().numpy().reshape(-1, 1)
                
                # Predict probability
                norm_loss_map_clean = np.nan_to_num(norm_loss_map, nan=0.0, posinf=1.0, neginf=0.0)
                probs = self.ema_gmm.predict_proba(norm_loss_map_clean)
                trust_prob = probs[:, self._trust_idx].reshape(batch_size, loss_map.shape[1], loss_map.shape[2])
                trust_prob = torch.tensor(trust_prob, device=dev)
                
                # Masks
                trust_mask = (trust_prob > 0.70).float()
                unc_mask = 1.0 - trust_mask

            # 2) Extract target features for perturbation (we use encode_decode inner backbone feature)
            # Since encode_decode doesn't give features easily without rewriting decode_head, 
            # we'll extract directly and perturb the final feature map.
            features = self.get_model().extract_feat(target_img)
            last_feat = features[-1].clone().detach() # [B, C, H', W']
            
            # Adversarial step parameters
            d = torch.randn_like(last_feat) * 1e-3
            d = d / (torch.norm(d, dim=1, keepdim=True) + 1e-8)
            
            for _ in range(self.K_steps):
                x_d = last_feat + d
                x_d.requires_grad = True
                
                # Replace features[-1] in decode head wrapper manually if supported, or just forward decode_head
                # Detach earlier features so we don't build graph back into the backbone
                mod_features = [f.detach() for f in features[:-1]] + [x_d]
                
                if hasattr(self.get_model(), 'decode_head'):
                    t_d_logit = self.get_model().decode_head.forward(mod_features)
                else: 
                    break # Safety fallback
                
                t_d_logit = F.interpolate(t_d_logit, size=target_img.shape[2:], mode='bilinear', align_corners=False)
                
                # Adversarial loss: Maximize CE for trustworthy, maximize KL for uncertain
                t_d_log_smax = torch.log_softmax(t_d_logit, dim=1)
                
                ce_spatial = self.pixel_ce(t_d_logit, pseudo_label)
                # spatially mask it
                adv_obj_ce = (ce_spatial * trust_mask).mean()
                
                # For uncertain, maximize KL divergence 
                # (We treat original t_logits as target for KL)
                adv_obj_kl = (F.kl_div(t_d_log_smax, ema_smax.detach(), reduction='none').sum(1) * unc_mask).mean()
                
                adv_obj = adv_obj_ce + adv_obj_kl
                g = torch.autograd.grad(adv_obj, x_d)[0]
                
                d = g.detach()
                d = d / (torch.norm(d, dim=1, keepdim=True) + 1e-8)
            
            # Final forward with perturbed features
            x_adv = last_feat + d  # No requires_grad
            mod_features = list(features)
            mod_features[-1] = x_adv
            if hasattr(self.get_model(), 'decode_head'):
                adv_logit = self.get_model().decode_head.forward(mod_features)
                adv_logit = F.interpolate(adv_logit, size=target_img.shape[2:], mode='bilinear', align_corners=False)
                
                # Regularization loss
                loss_ce_trust = (self.pixel_ce(adv_logit, pseudo_label) * trust_mask).mean() * 0.3
                
                adv_smax = torch.log_softmax(adv_logit, dim=1)
                loss_kl_unc = (F.kl_div(adv_smax, ema_smax.detach(), reduction='none').sum(1) * unc_mask).mean() * 0.1
                
                tot_adv_loss = loss_ce_trust + loss_kl_unc
                log_vars['loss_pefat_adv'] = tot_adv_loss.item()
                tot_adv_loss.backward()

            if self.local_iter % self.debug_img_interval == 0:
                out_dir = os.path.join(self.train_cfg['work_dir'], 'pefat_debug')
                os.makedirs(out_dir, exist_ok=True)
                vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
                vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
                pred_adv = adv_logit.argmax(1)
                pred_ema = ema_smax.argmax(1)
                for j in range(batch_size):
                    rows, cols = 2, 4
                    fig, axs = plt.subplots(
                        rows,
                        cols,
                        figsize=(3 * cols, 3 * rows),
                        gridspec_kw={
                            'hspace': 0.1,
                            'wspace': 0,
                            'top': 0.95,
                            'bottom': 0,
                            'right': 1,
                            'left': 0
                        },
                    )
                    subplotimg(axs[0][0], vis_img[j], 'Source Image')
                    subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
                    subplotimg(axs[0][1], gt_semantic_seg[j], 'Source GT', cmap='cityscapes')
                    subplotimg(axs[1][1], pseudo_label[j], 'Pseudo GT', cmap='cityscapes')
                    subplotimg(axs[0][2], trust_mask[j], 'Trust Mask', cmap='gray')
                    subplotimg(axs[1][2], unc_mask[j], 'Unc Mask', cmap='gray')
                    subplotimg(axs[0][3], pred_adv[j], 'Adv Pred', cmap='cityscapes')
                    subplotimg(axs[1][3], pred_ema[j], 'EMA Pred', cmap='cityscapes')
                    for ax in axs.flat:
                        ax.axis('off')
                    plt.savefig(
                        os.path.join(out_dir,
                                        f'{(self.local_iter + 1):06d}_{j}.png'))
                    plt.close()
        self.local_iter += 1
        return log_vars

