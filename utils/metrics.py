import numpy as np
import torch
import math
from torch.autograd import Variable
from scipy.stats import norm
import properscoring as ps


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    return mae, mse, rmse, mape, mspe


'''
Evauation metrics
'''


def prob_loss(mu: Variable, sigma: Variable, labels: Variable):
    '''
    Compute using gaussian the log-likehood which needs to be maximized.
    Args:
        mu: (Variable) dimension [batch_size] - estimated mean at time step t
        sigma: (Variable) dimension [batch_size] - estimated standard deviation at time step t
        labels: (Variable) dimension [batch_size] z_t
    Returns:
        loss: (Variable) average log-likelihood loss across the batch
    '''
    distribution = torch.distributions.normal.Normal(mu, sigma)
    likelihood = distribution.log_prob(labels)
    return -torch.mean(likelihood)


# if relative is set to True, metrics are not normalized by the scale of labels
def accuracy_ND(mu: torch.Tensor, labels: torch.Tensor, relative=False):
    zero_index = (labels != 0)
    if relative:
        diff = torch.mean(torch.abs(mu[zero_index] - labels[zero_index])).item()
        return [diff, 1]
    else:
        diff = torch.sum(torch.abs(mu[zero_index] - labels[zero_index])).item()
        summation = torch.sum(torch.abs(labels[zero_index])).item()
        return [diff, summation]


def accuracy_RMSE(mu: torch.Tensor, labels: torch.Tensor, relative=False):
    zero_index = (labels != 0)
    diff = torch.sum(torch.mul((mu[zero_index] - labels[zero_index]), (mu[zero_index] - labels[zero_index]))).item()
    if relative:
        return [diff, torch.sum(zero_index).item(), torch.sum(zero_index).item()]
    else:
        summation = torch.sum(torch.abs(labels[zero_index])).item()
        if summation == 0:
            logger.error('summation denominator error! ')
        return [diff, summation, torch.sum(zero_index).item()]


def accuracy_ROU(rou: float, samples: torch.Tensor, labels: torch.Tensor, relative=False):
    numerator = 0
    denominator = 0
    pred_samples = samples.shape[0]
    for t in range(labels.shape[1]):
        zero_index = (labels[:, t] != 0)
        if zero_index.numel() > 0:
            rou_th = math.ceil(pred_samples * (1 - rou))
            rou_pred = torch.topk(samples[:, zero_index, t], dim=0, k=rou_th)[0][-1, :]
            abs_diff = labels[:, t][zero_index] - rou_pred
            numerator += 2 * (torch.sum(rou * abs_diff[labels[:, t][zero_index] > rou_pred]) - torch.sum(
                (1 - rou) * abs_diff[labels[:, t][zero_index] <= rou_pred])).item()
            denominator += torch.sum(labels[:, t][zero_index]).item()
    if relative:
        return [numerator, torch.sum(labels != 0).item()]
    else:
        return [numerator, denominator]


def accuracy_ND_(mu: torch.Tensor, labels: torch.Tensor, relative=False):
    mu = mu.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    mu[labels == 0] = 0.

    diff = np.sum(np.abs(mu - labels), axis=1)
    if relative:
        summation = np.sum((labels != 0), axis=1)
        mask = (summation == 0)
        summation[mask] = 1
        result = diff / summation
        result[mask] = -1
        return result
    else:
        summation = np.sum(np.abs(labels), axis=1)
        mask = (summation == 0)
        summation[mask] = 1
        result = diff / summation
        result[mask] = -1
        return result


def accuracy_RMSE_(mu: torch.Tensor, labels: torch.Tensor, relative=False):
    mu = mu.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    mask = labels == 0
    mu[mask] = 0.

    diff = np.sum((mu - labels) ** 2, axis=1)
    summation = np.sum(np.abs(labels), axis=1)
    mask2 = (summation == 0)
    if relative:
        div = np.sum(~mask, axis=1)
        div[mask2] = 1
        result = np.sqrt(diff / div)
        result[mask2] = -1
        return result
    else:
        summation[mask2] = 1
        result = (np.sqrt(diff) / summation) * np.sqrt(np.sum(~mask, axis=1))
        result[mask2] = -1
        return result


def accuracy_ROU_(rou: float, samples: torch.Tensor, labels: torch.Tensor, relative=False):
    samples = samples.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    mask = labels == 0
    samples[:, mask] = 0.

    pred_samples = samples.shape[0]
    rou_th = math.floor(pred_samples * rou)

    samples = np.sort(samples, axis=0)
    rou_pred = samples[rou_th]

    abs_diff = np.abs(labels - rou_pred)
    abs_diff_1 = abs_diff.copy()
    abs_diff_1[labels < rou_pred] = 0.
    abs_diff_2 = abs_diff.copy()
    abs_diff_2[labels >= rou_pred] = 0.

    numerator = 2 * (rou * np.sum(abs_diff_1, axis=1) + (1 - rou) * np.sum(abs_diff_2, axis=1))
    denominator = np.sum(labels, axis=1)

    mask2 = (denominator == 0)
    denominator[mask2] = 1
    result = numerator / denominator
    result[mask2] = -1
    return result


def init_metrics(sample=True):
    metrics = {
        'ND': np.zeros(2),  # numerator, denominator
        'RMSE': np.zeros(3),  # numerator, denominator, time step count
        'test_loss': np.zeros(2),
    }
    if sample:
        metrics['rou90'] = np.zeros(2)
        metrics['rou50'] = np.zeros(2)
    return metrics


def get_metrics(sample_mu, labels, predict_start, samples=None, relative=False):
    prob_metric = dict()
    prob_metric['ND'] = accuracy_ND_(sample_mu, labels[:, predict_start:], relative=relative)
    prob_metric['RMSE'] = accuracy_RMSE_(sample_mu, labels[:, predict_start:], relative=relative)
    if samples is not None:
        prob_metric['rou90'] = accuracy_ROU_(0.9, samples, labels[:, predict_start:], relative=relative)
        prob_metric['rou50'] = accuracy_ROU_(0.5, samples, labels[:, predict_start:], relative=relative)
    return prob_metric


def update_metrics(raw_metrics, input_mu, input_sigma, sample_mu, labels, predict_start, samples=None, relative=False):
    raw_metrics['ND'] = raw_metrics['ND'] + accuracy_ND(sample_mu, labels[:, predict_start:], relative=relative)
    raw_metrics['RMSE'] = raw_metrics['RMSE'] + accuracy_RMSE(sample_mu, labels[:, predict_start:], relative=relative)
    input_time_steps = input_mu.numel()
    raw_metrics['test_loss'] = raw_metrics['test_loss'] + [
        loss_fn(input_mu, input_sigma, labels[:, :predict_start]) * input_time_steps, input_time_steps]
    if samples is not None:
        raw_metrics['rou90'] = raw_metrics['rou90'] + accuracy_ROU(0.9, samples, labels[:, predict_start:],
                                                                   relative=relative)
        raw_metrics['rou50'] = raw_metrics['rou50'] + accuracy_ROU(0.5, samples, labels[:, predict_start:],
                                                                   relative=relative)
    return raw_metrics


def final_metrics(raw_metrics, sampling=False):
    summary_metric = dict()
    summary_metric['ND'] = raw_metrics['ND'][0] / raw_metrics['ND'][1]
    summary_metric['RMSE'] = np.sqrt(raw_metrics['RMSE'][0] / raw_metrics['RMSE'][2]) / (
            raw_metrics['RMSE'][1] / raw_metrics['RMSE'][2])
    summary_metric['test_loss'] = (raw_metrics['test_loss'][0] / raw_metrics['test_loss'][1]).item()
    if sampling:
        summary_metric['rou90'] = raw_metrics['rou90'][0] / raw_metrics['rou90'][1]
        summary_metric['rou50'] = raw_metrics['rou50'][0] / raw_metrics['rou50'][1]
    return summary_metric


"""
Function for evaluation
"""


def SMAPE(yPred, yTrue):
    assert len(yPred) == len(yTrue)
    denominator = (np.abs(yTrue) + np.abs(yPred))
    diff = np.abs(yTrue - yPred) / denominator
    diff[denominator == 0] = 0.0
    return 200 * np.mean(diff)


def ND(yPred, yTrue):  # normalized deviation
    assert len(yPred) == len(yTrue)
    demoninator = np.sum(np.abs(yTrue))
    diff = np.sum(np.abs(yTrue - yPred))
    return 1.0 * diff / demoninator


def RMSLE(yPred, yTrue):
    assert len(yPred) == len(yTrue)
    assert len(yTrue) == len(yPred)
    return np.sqrt(np.mean((np.log(1 + yPred) - np.log(1 + yTrue)) ** 2))


def NRMSE(yPred, yTrue):
    assert len(yPred) == len(yTrue)
    denominator = np.mean(yTrue)
    diff = np.sqrt(np.mean(((yPred - yTrue) ** 2)))
    return diff / denominator


def rhoRisk2(yPred, yTrue, rho):
    assert len(yPred) == len(yTrue)
    diff1 = (yTrue - yPred) * rho * (yTrue >= yPred)
    diff2 = (yPred - yTrue) * (1 - rho) * (yTrue < yPred)
    denominator = np.sum(yTrue)
    return 2 * (np.sum(diff1) + np.sum(diff2)) / denominator


def quantile_loss(pred, true, rho):
    loss = np.where(true >= pred, rho * (np.abs(true - pred)), (1 - rho) * (np.abs(true - pred)))  # positive
    return np.sum(loss)


def rhoRisk(yPred, yTrue, rho):  # normalized sum of quantile losses
    assert len(yPred) == len(yTrue)
    # diff = -np.sum((yPred-yTrue)*(rho*(yPred <= yTrue)-(1-rho)*(yPred > yTrue)))
    # denominator = np.sum(yTrue)
    # return diff/denominator
    ql = quantile_loss(yPred, yTrue, rho)
    return ql / np.sum(yTrue)


def probEvaluator(mu, sigma, yTrue):
    validPredQ50 = mu
    validPredQ90 = norm.ppf(0.9, mu, sigma)
    validTrue = yTrue
    # The evaluation metrics
    rho50 = rhoRisk(validPredQ50.reshape(-1, ), validTrue.reshape(-1, ), 0.5)
    rho90 = rhoRisk(validPredQ90.reshape(-1, ), validTrue.reshape(-1, ), 0.9)
    mae = MAE(mu, yTrue)
    rmse = RMSE(mu, yTrue)
    crps = ps.crps_gaussian(yTrue, mu=mu, sig=sigma).mean()
    return crps, rho50, rho90, mae, rmse
