import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, LogisticRegression
from lib.utils import GeneralizedCELoss
from lib import joint_dro
from fairlearn.reductions import DemographicParity, EqualizedOdds, ErrorRateParity, ExponentiatedGradient

class ERM(nn.Module):

    def __init__(self, hparams, task='classification', es_patience=3,
                 val_pct=0.2):
        super().__init__()
        self.hparams = hparams
        self.task = task
        self.lr = hparams['lr']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = hparams['batch_size']  # if None, use full batch
        self.C = hparams['C']
        self.es_patience = es_patience
        self.val_pct = val_pct
        self.debug = hparams['debug']
        self.max_epochs = hparams['max_epochs']

    def init_network(self):
        self.network = nn.Linear(self.n_features, 1).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.lr
        )

    def set_train(self):
        self.network.train()

    def set_eval(self):
        self.network.eval()

    def fit(self, X, y, sample_weight, grp):
        X = torch.tensor(X).float()
        y = torch.tensor(y).float().squeeze()
        self.unique_groups = np.unique(grp)
        grp = torch.tensor(grp).float().squeeze()       

        if self.hparams['ignore_lime_weights']:
            sample_weight = torch.ones(*sample_weight.shape).float().squeeze()
        else:
            sample_weight = torch.tensor(sample_weight).float().squeeze()

        if self.batch_size is None:
            self.batch_size = X.shape[0]

        self.fitted = True
        self.n_features = X.shape[1]
        self.init_network()

        if self.task == 'classification':
            self.loss_func_erm = lambda X, y, sample_weight: torch.dot(F.binary_cross_entropy_with_logits(
                self.network(X).squeeze(-1), y, reduction='none'), sample_weight)
        elif self.task == 'regression':
            self.loss_func_erm = lambda X, y, sample_weight: torch.dot(
                F.mse_loss(self.network(X).squeeze(-1), y, reduction='none'), sample_weight)
        else:
            raise NotImplementedError(self.task)

        train_inds = np.random.permutation(
            len(X))[:int(len(X) * (1 - self.val_pct))]
        val_inds = np.setdiff1d(np.arange(len(X)), train_inds)

        train_ds = torch.utils.data.TensorDataset(
            X[train_inds], y[train_inds], sample_weight[train_inds], grp[train_inds])
        val_ds = torch.utils.data.TensorDataset(
            X[val_inds], y[val_inds], sample_weight[val_inds], grp[val_inds])

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,
                                                   num_workers=1)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=self.batch_size, shuffle=False,
                                                 num_workers=1)

        best_loss, best_state = None, None
        es_counter = 0
        epoch = 0

        while es_counter < self.es_patience and epoch < self.max_epochs:
            self.set_train()
            for train_X, train_y, train_sample_weight, train_g in train_loader:
                train_X, train_y, train_sample_weight, train_g = train_X.to(self.device), train_y.to(
                    self.device), train_sample_weight.to(self.device), train_g.to(self.device)
                res_i = self.train_batch(train_X, train_y,
                                         train_sample_weight, train_g)
                if self.debug:
                    print(res_i)

            self.set_eval()
            val_loss = 0
            with torch.no_grad():
                for val_X, val_y, val_sample_weight, _ in val_loader:
                    val_X, val_y, val_sample_weight = val_X.to(self.device), val_y.to(
                        self.device), val_sample_weight.to(self.device)
                    val_loss += self.loss_func_erm(val_X,
                                                   val_y, val_sample_weight)

            if self.debug:
                print(val_loss)
            if best_loss is None or val_loss < best_loss:
                best_loss = val_loss
                best_state = self.network.state_dict()
                es_counter = 0
            else:
                es_counter += 1
            epoch += 1

        self.network.load_state_dict(best_state)
        self.set_eval()
        self.to('cpu')

    def train_batch(self, X, y, sample_weight, *args, **kwargs):
        loss = self.loss_func_erm(X, y, sample_weight) + \
            self.C * torch.norm(self.network.weight)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, X):
        assert self.fitted
        self.set_eval()
        with torch.no_grad():
            y_pred = self.network(torch.tensor(X).float().to(
                self.network.weight.device)).squeeze(-1)
        if self.task == 'classification':
            y_pred = torch.sigmoid(y_pred)
        return y_pred.detach().cpu().numpy()

    def score(self, X, y, sample_weight):
        self.set_eval()
        with torch.no_grad():
            return r2_score(y_true=y, y_pred=self.predict(X), sample_weight=sample_weight)

    @property
    def intercept_(self):
        assert self.fitted
        return self.network.bias.detach().cpu().double().numpy().squeeze()

    @property
    def coef_(self):
        assert self.fitted
        return self.network.weight.detach().cpu().double().numpy().squeeze()


class ARL(ERM):
    '''
    Adversarially reweighted learning
    https://arxiv.org/pdf/2006.13114.pdf
    '''

    def init_network(self):
        self.network = nn.Linear(self.n_features, 1).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.lr
        )

        self.discriminator = nn.Linear(self.n_features + 1, 1).to(self.device)
        self.disc_optimizer = torch.optim.Adam(
            list(self.discriminator.parameters()),
            lr=self.lr
        )

    def set_train(self):
        self.network.train()
        self.discriminator.train()

    def set_eval(self):
        self.network.eval()
        self.discriminator.eval()

    def train_batch(self, X, y, sample_weight, *args, **kwargs):
        if self.task == 'classification':
            clf_losses = F.binary_cross_entropy_with_logits(
                self.network(X).squeeze(), y, reduction='none')
        elif self.task == 'regression':
            clf_losses = F.mse_loss(self.network(
                X).squeeze(), y, reduction='none')

        disc_input = torch.cat((X, y.unsqueeze(-1)), dim=-1)
        disc_out = torch.sigmoid(self.discriminator(disc_input))
        disc_weights = 1 + len(disc_out) * (disc_out / disc_out.sum())
        clf_loss_normal = torch.dot(
            clf_losses, sample_weight) + self.C * torch.norm(self.network.weight)
        clf_loss_weighted = torch.dot(clf_losses, disc_weights.squeeze(
        ) * sample_weight.squeeze()) + self.C * torch.norm(self.network.weight)

        self.optimizer.zero_grad()
        self.disc_optimizer.zero_grad()

        clf_loss_weighted.backward()
        for param in self.disc_optimizer.param_groups[0]['params']:
            param.grad *= -1

        self.optimizer.step()
        self.disc_optimizer.step()

        return {'loss': clf_loss_weighted.item(), 'loss_normal': clf_loss_normal.item(),
                'max_weight': disc_weights.max().item(), 'min_weight': disc_weights.min().item()}


class JTT():
    '''
    Just Train Twice
    https://arxiv.org/pdf/2107.09044.pdf
    '''

    def __init__(self, hparams):
        self.hparams = hparams
        self.thres = hparams['jtt_thres']
        self.lamb = hparams['jtt_lambda']
        self.C = hparams['C']

    def fit(self, X, y, sample_weight, grp):
        if self.hparams['ignore_lime_weights']:
            sample_weight = np.ones(shape=sample_weight.shape)

        # we use a regressor here since y is in [0, 1] - this is what LIME does
        m1 = Ridge(alpha=self.C).fit(X, y, sample_weight=sample_weight)
        
        # the points which were predicted correctly, won't be upweighted
        m1_pred = np.array(m1.predict(X) >= self.thres, dtype=np.int8)==np.array(y >= self.thres, dtype=np.int8)
        new_weights = np.where(m1_pred, np.ones(
            X.shape[0]), np.ones(X.shape[0]) * self.lamb)
        
        # NB: Normalizing weights
        # Empirically, this does not change results in comparison to just
        # multiplying both
        new_normalized_weights=np.multiply(new_weights, sample_weight)
        new_normalized_weights=new_normalized_weights/np.sum(new_normalized_weights)

        m2 = Ridge(alpha=self.C).fit(
            X, y, sample_weight=new_normalized_weights)
        self.model = m2

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y, sample_weight):
        return r2_score(y_true=y, y_pred=self.predict(X), sample_weight=sample_weight)

    @property
    def intercept_(self):
        return self.model.intercept_

    @property
    def coef_(self):
        return self.model.coef_

class JointDRO(ERM): 
    '''
    Large-Scale Methods for Distributionally Robust Optimization
    (CVaR or joint DRO)
    https://arxiv.org/pdf/2010.05893.pdf
    '''
    
    def __init__(self, hparams):
        super().__init__(hparams)
        self.hparams = hparams
        self.alpha = hparams['joint_dro_alpha']
        self.C = hparams['C']
        #size, reg, geometry
        self._joint_dro_loss_computer = joint_dro.RobustLoss(size=hparams['joint_dro_alpha'], 
                reg=0, 
                geometry="cvar")

    def train_batch(self, X, y, sample_weight, grp=None):
        if self.task == 'classification':
            per_sample_losses = F.binary_cross_entropy_with_logits(
                self.network(X).squeeze(), y, reduction='none')
        elif self.task == 'regression':
            per_sample_losses = F.mse_loss(
                self.network(X).squeeze(), y, reduction='none')
        
        loss = self._joint_dro_loss_computer(per_sample_losses) + \
            self.C * torch.norm(self.network.weight)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

class LfF(ERM):
    '''
    Learning from Failure
    https://arxiv.org/pdf/2007.02561.pdf
    '''

    def init_network(self):
        super().init_network()
        self.biased = nn.Linear(self.n_features, 1).to(self.device)
        self.biased_optimizer = torch.optim.Adam(
            self.biased.parameters(),
            lr=self.lr
        )

        self.network = nn.Linear(self.n_features, 1).to(self.device)
        self.debiased_optimizer = torch.optim.Adam(
            list(self.network.parameters()),
            lr=self.lr
        )

    def set_train(self):
        self.network.train()
        self.biased.train()

    def set_eval(self):
        self.network.eval()
        self.biased.eval()

    def train_batch(self, X, y, sample_weight, *args, **kwargs):
        if self.task == 'classification':
            # define loss
            criterion = nn.CrossEntropyLoss(reduction='none')
            bias_criterion = GeneralizedCELoss()
        elif self.task == 'regression':
            raise NotImplementedError
        y=y.long()
        logit_biased = self.biased(X)
        if np.isnan(logit_biased.mean().item()):
            raise NameError('logit_biased')
        logit_debiased = self.network(X)

        loss_b = criterion(logit_biased, y).cpu().detach()
        loss_d = criterion(logit_debiased, y).cpu().detach()

        if np.isnan(loss_b.mean().item()):
            raise NameError('loss_b')
        if np.isnan(loss_d.mean().item()):
            raise NameError('loss_d')

        # re-weighting based on loss value / generalized CE for biased modeli
        loss_weight = loss_b / (loss_b + loss_d + 1e-8)
        loss_weight = torch.tensor(sample_weight).to(self.device)*loss_weight.to(self.device)
        if np.isnan(loss_weight.mean().item()):
            raise NameError('loss_weight')

        loss_b_update = bias_criterion(logit_biased, y)
        loss_b_updated = loss_b_update * torch.tensor(sample_weight).to(self.device)
        if np.isnan(loss_b_update.mean().item()):
            raise NameError('loss_b_update')

        loss_d_update = criterion(logit_debiased, y) * \
            loss_weight.to(self.device)+ \
            self.C * torch.norm(self.network.weight)
        if np.isnan(loss_d_update.mean().item()):
            raise NameError('loss_d_update')

        loss = loss_b_update.mean() + loss_d_update.mean()

        self.biased_optimizer.zero_grad()
        self.debiased_optimizer.zero_grad()
        loss.backward()
        self.biased_optimizer.step()
        self.debiased_optimizer.step()

        return {'loss': loss.item(), 'loss_b': loss_b_update.mean().item(),
                'loss_d': loss_d_update.mean().item()}

class GroupDRO(ERM):
    '''
    Group Distributionally Robust Optimization
    https://arxiv.org/abs/1911.08731
    '''
    def init_network(self):
        super().init_network()
        self.register_buffer("q", torch.ones(len(self.unique_groups)))
        self.mapping = {i:c for c, i in enumerate(self.unique_groups)}

    def train_batch(self, X, y, sample_weight, grp):
        if str(self.q.device) != self.device:
            self.q = self.q.to(self.device)

        grps_in_minibatch = torch.unique(grp)
        losses = torch.zeros(len(self.unique_groups)).to(self.device)
        for grp_m in grps_in_minibatch:
            mask = grp == grp_m
            x_m, y_m, w_m = X[mask], y[mask], sample_weight[mask]
            losses[self.mapping[grp_m.item()]] = self.loss_func_erm(x_m, y_m, w_m) / len(y_m)
            self.q[self.mapping[grp_m.item()]] *= (self.hparams['groupdro_eta'] * losses[self.mapping[grp_m.item()]].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q) / len(grps_in_minibatch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item(), 'weights': self.q.detach().cpu().numpy()}

class Reductionist():
    '''
    A Reductions Approach to Fair Classification
    https://arxiv.org/pdf/1803.02453.pdf    
    '''
    def __init__(self, hparams):
        self.hparams = hparams
        self.reductionist_type = hparams['reductionist_type']
        self.reductionist_difference_bound = hparams['reductionist_difference_bound']
        self.C = hparams['C']
        self.thres = hparams['reductionist_thres']
        
    def fit(self, X, y, sample_weight, grp):
        assert(self.hparams['ignore_lime_weights'])

        if self.reductionist_type == 'DP':
            cons = DemographicParity(difference_bound=self.reductionist_difference_bound)
        elif self.reductionist_type == 'EO':
            cons = EqualizedOdds(difference_bound=self.reductionist_difference_bound)
        elif self.reductionist_type == 'Acc':
            cons = ErrorRateParity(difference_bound=self.reductionist_difference_bound)
        else:
            raise NotImplementedError

        y = y >= self.thres # requires binary labels
        est = LogisticRegression(C=self.C)
        eg = ExponentiatedGradient(est, cons, eps = self.reductionist_difference_bound).fit(X, y, sensitive_features = grp)
        self.model = eg.predictors_.iloc[-1]

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]

    def score(self, X, y, sample_weight):
        return r2_score(y_true=y, y_pred=self.model.predict_proba(X)[:, 1], sample_weight=sample_weight)

    @property
    def intercept_(self):
        return self.model.intercept_

    @property
    def coef_(self):
        return self.model.coef_
