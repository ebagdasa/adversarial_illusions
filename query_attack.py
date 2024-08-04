import sys
from pathlib import Path

import toml

import numpy as np

import torch
from torch.utils.data import DataLoader

from utils import norm, unnorm, criterion
from dataset_utils import create_dataset
from models import load_model
import time


# Configure Script
config = toml.load(f'configs/imagenet/query/{sys.argv[1]}.toml')['general']
epochs = config['epochs']
gpu_num = config['gpu_num']
batch_size = config['batch_size']
eps = config['epsilon']
seed = config['seed']
output_dir = config['output_dir']
n_images = config['number_images']
buffer_size = config['buffer_size']
delta = config['delta']
model_flag = config.get('model_flag', 'imagebind')
embs_input = config.get('embeddings_input', output_dir + 'embs.npy')\
                   .format(model_flag)
modality = config.get('modality', 'vision')
dataset_flag = config.get('dataset_flag', 'imagenet')
input_images_file = config.get('input_images_file', None)

if modality == 'vision':
    eps = eps / 255
    
Path(output_dir).mkdir(parents=True, exist_ok=True)

full_flag=False
if "full" in output_dir:
    full_flag = True

print('Full_flag: ',full_flag)

hybrid=False
if input_images_file!=None:
    hybrid=True
print('Hybrid: ',hybrid)

device = f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu"
assert n_images % batch_size == 0

# Instantiate Model
model = load_model(model_flag, device)

# Load Data
image_text_dataset = create_dataset(dataset_flag, model=model, device=device, seed=seed, embs_input=embs_input)
# Create Adversarial Examples
X_advs = []
X_inits = []
gts = []
gt_loss = []
adv_loss = []
end_iter = []

# TODO: verify added code
y_ids = []
y_origs = []

final = []

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)
    
def get_loss(y, logits, targeted=False, loss_type='margin_loss'):
    """ Implements the margin loss (difference between the correct and 2nd best class). """
    if loss_type == 'margin_loss':
        preds_correct_class = (logits * y).sum(1, keepdims=True)
        diff = preds_correct_class - logits  # difference between the correct class and all other classes
        diff[y] = np.inf  # to exclude zeros coming from f_correct - f_correct
        margin = diff.min(1, keepdims=True)
        loss = margin * -1 if targeted else margin
        # print(loss)
    elif loss_type == 'cross_entropy':
        probs = softmax(logits)
        # print(y)
        # print(probs.shape)
        loss = -np.log(probs[y])
        loss = loss * -1 if not targeted else loss
    return loss.flatten()
    
def dense_to_onehot(y_test, n_cls):
    y_test_onehot = np.zeros([len(y_test), n_cls], dtype=bool)
    y_test_onehot[np.arange(len(y_test)), y_test] = True
    return y_test_onehot

def p_selection(p_init, it, n_iters):
    """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
    it = int(it / n_iters * 10000)

    if 10 < it <= 50:
        p = p_init / 2
    elif 50 < it <= 200:
        p = p_init / 4
    elif 200 < it <= 500:
        p = p_init / 8
    elif 500 < it <= 1000:
        p = p_init / 16
    elif 1000 < it <= 2000:
        p = p_init / 32
    elif 2000 < it <= 4000:
        p = p_init / 64
    elif 4000 < it <= 6000:
        p = p_init / 128
    elif 6000 < it <= 8000:
        p = p_init / 256
    elif 8000 < it <= 10000:
        p = p_init / 512
    else:
        p = p_init

    return p

def square_attack_linf(model, x, y, eps, n_iters, p_init, metrics_path, targeted, loss_type, local_adv=None):
    """ The Linf square attack """
    np.random.seed(0)  # important to leave it here as well
    early_break=False
    x=unnorm(x).to(device)
    min_val, max_val = 0, 1
    c, h, w = x.shape[1:]
    n_features = c*h*w
    n_ex_total = x.shape[0]

    if local_adv==None:
        init_delta = torch.tensor(np.random.choice([-eps, eps], size=[x.shape[0], c, 1, w])).to(torch.float).to(device)
    else:
        local_adv=unnorm(local_adv).to(device)
        init_delta = local_adv-x
    # init_delta=0
    x_best = torch.clip(x + init_delta, min_val, max_val)
    with torch.no_grad():
        embeds = model.forward(x_best.cuda(), modality, normalize=True)
    logits=criterion(embeds[:, None, :].cpu(), image_text_dataset.labels[None, :, :].cpu(), dim=2).detach().cpu().numpy()
    loss_min = get_loss(y, logits, targeted, loss_type=loss_type)
    margin_min = get_loss(y, logits, targeted, loss_type='margin_loss')
    n_queries = np.ones(x.shape[0])  # ones because we have already used 1 query

    time_start = time.time()
    metrics = np.zeros([n_iters, 7])
    for i_iter in range(n_iters - 1):
        idx_to_fool = margin_min > 0
        x_curr, x_best_curr, y_curr = x[idx_to_fool], x_best[idx_to_fool], y[idx_to_fool]
        loss_min_curr, margin_min_curr = loss_min[idx_to_fool], margin_min[idx_to_fool]
        deltas = x_best_curr - x_curr
        p = p_selection(p_init, i_iter, n_iters)
        for i_img in range(x_best_curr.shape[0]):
            s = int(round(np.sqrt(p * n_features / c)))
            s = min(max(s, 1), h-1)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
            center_h = np.random.randint(0, h - s)
            center_w = np.random.randint(0, w - s)

            x_curr_window = x_curr[i_img, :, center_h:center_h+s, center_w:center_w+s]
            x_best_curr_window = x_best_curr[i_img, :, center_h:center_h+s, center_w:center_w+s]
            # prevent trying out a delta if it doesn't change x_curr (e.g. an overlapping patch)
            while torch.sum(torch.abs(torch.clip(x_curr_window + deltas[i_img, :, center_h:center_h+s, center_w:center_w+s], min_val, max_val) - x_best_curr_window) < 10**-7) == c*s*s:
                deltas[i_img, :, center_h:center_h+s, center_w:center_w+s] = torch.tensor(np.random.choice([-eps, eps], size=[c, 1, 1]))

        x_new = torch.clip(x_curr + deltas, min_val, max_val).to(device)
        with torch.no_grad():
            embeds = model.forward(x_new, modality, normalize=True)
        logits=criterion(embeds[:, None, :].cpu(), image_text_dataset.labels[None, :, :].cpu(), dim=2).detach().cpu().numpy()
        loss = get_loss(y_curr, logits, targeted, loss_type=loss_type)
        margin = get_loss(y_curr, logits, targeted, loss_type='margin_loss')

        
        idx_improved = loss < loss_min_curr
        loss_min[idx_to_fool] = idx_improved * loss + ~idx_improved * loss_min_curr
        margin_min[idx_to_fool] = idx_improved * margin + ~idx_improved * margin_min_curr
        idx_improved = torch.tensor(np.reshape(idx_improved, [-1, *[1]*len(x.shape[:-1])])).to(device)
        x_best[idx_to_fool] = idx_improved * x_new + ~idx_improved * x_best_curr
        n_queries[idx_to_fool] += 1
        acc = (margin_min > 0.0).sum() / n_ex_total
        acc_corr = (margin_min > 0.0).mean()
        mean_nq, mean_nq_ae, median_nq_ae = np.mean(n_queries), np.mean(n_queries[margin_min <= 0]), np.median(n_queries[margin_min <= 0])
        avg_margin_min = np.mean(margin_min)
        time_total = time.time() - time_start
        if i_iter%10==0:
            print('{}: acc={:.2%} acc_corr={:.2%} avg#q_ae={:.2f} med#q={:.1f}, avg_margin={:.6f} (n_ex={}, eps={:.3f}, {:.2f}s)'.
                format(i_iter+1, acc, acc_corr, mean_nq_ae, median_nq_ae, avg_margin_min, x.shape[0], eps, time_total))

        metrics[i_iter] = [acc, acc_corr, mean_nq, mean_nq_ae, median_nq_ae, margin_min.mean(), time_total]
            
        if i_iter>10000 and (metrics[i_iter-10000][5]-metrics[i_iter][5]<0.00001):
            early_break=True
        if (i_iter <= 500 and i_iter % 20 == 0) or (i_iter > 100 and i_iter % 50 == 0) or i_iter + 1 == n_iters or acc == 0 or early_break==True:
            np.save(metrics_path, metrics)

        if full_flag==False:
            if acc == 0 or early_break==True:
                break
        else:
            if early_break==True:
                break

    return n_queries, x_best

targeted=True  
p=0.05
if targeted:
    p=0.01
loss='cross_entropy'
torch.manual_seed(seed)
if hybrid==True:
    local_advs_path = input_images_file
    local_advs = np.load(local_advs_path)
    local_advs=torch.tensor(local_advs).to(device)

    success=0
    n=0
    dataloader = DataLoader(image_text_dataset, batch_size=1, shuffle=True)
    for i, (X, Y, gt, y_id, y_orig) in enumerate(dataloader):
        if n==100:
            break
        embeds = model.forward(local_advs[i].unsqueeze(0), modality, normalize=False)
        classes = criterion(embeds[:, None, :].cpu(), image_text_dataset.labels[None, :, :].cpu(), dim=2).argsort(dim=1, descending=True)
        if torch.all((classes == y_id[:, None]).nonzero(as_tuple=True)[1].cpu() == 0):
            success+=1
        n+=1
    print("the attack success rate is "+ str(success)+ "%")
    print("-----------------------------------------")

torch.manual_seed(seed)
dataloader = DataLoader(image_text_dataset, batch_size=batch_size, shuffle=True)

for i, (X, Y, gt, y_id, y_orig) in enumerate(dataloader):
    metrics_path = f"{output_dir}metrics_{i}.npy"
    if targeted:
        y=dense_to_onehot(y_id, n_cls=1000)
    else:
        y=dense_to_onehot(y_orig, n_cls=1000)

    if hybrid==True:
        local_adv=local_advs[i*batch_size:i*batch_size+batch_size]
        n_queries, x_adv = square_attack_linf(model, X, y, eps, epochs,
                                        p, metrics_path, targeted, loss, local_adv)
    else:
        n_queries, x_adv = square_attack_linf(model, X, y, eps, epochs,
                                        p, metrics_path, targeted, loss)
    # Record batchwise information
    with torch.no_grad():
        norm_x_adv=norm(x_adv)
        gt_embeddings = model.forward(norm_x_adv.to(device), modality, normalize=True).detach().cpu()
        embeds = model.forward(x_adv.to(device), modality, normalize=True).detach().cpu()
        classes = criterion(embeds[:, None, :].cpu(), image_text_dataset.labels[None, :, :].detach().cpu(), dim=2).argsort(dim=1, descending=True)

    X_advs.append(norm_x_adv.detach().cpu().clone())
    X_inits.append(X.cpu().clone())
    gts.append(gt.cpu().clone())
    gt_loss.append(criterion(gt_embeddings, Y.cpu(), dim=1))
    adv_loss.append(criterion(embeds.detach().cpu(), Y.cpu(), dim=1))
    end_iter.append(n_queries)
    print((classes == y_id[:, None])[:, 0])
    y_ids.append(y_id.cpu())
    y_origs.append(y_orig.cpu())
    final.append((classes == y_id[:, None])[:, 0].cpu())
    
    if i == (n_images // batch_size)-1:
        break

np.save(output_dir + 'x_advs', np.concatenate(X_advs))
np.save(output_dir + 'x_inits', np.concatenate(X_inits))
np.save(output_dir + 'gts', np.concatenate(gts))
np.save(output_dir + 'gt_loss', np.concatenate(gt_loss))
np.save(output_dir + 'adv_loss', np.concatenate(adv_loss))
np.save(output_dir + 'end_iter', np.concatenate(end_iter))

np.save(output_dir + 'y_ids', np.concatenate(y_ids))
np.save(output_dir + 'y_origs', np.concatenate(y_origs))
np.save(output_dir + 'final', np.concatenate(final))

# Compute and print the average and standard deviation of gt_loss and adv_loss
gt_loss_avg = np.mean(np.concatenate(gt_loss))
gt_loss_std = np.std(np.concatenate(gt_loss))
adv_loss_avg = np.mean(np.concatenate(adv_loss))
adv_loss_std = np.std(np.concatenate(adv_loss))

print("Average organic alignment:", gt_loss_avg)
print("Standard deviation of organic alignment:", gt_loss_std)
print("Average adversarial alignment:", adv_loss_avg)
print("Standard deviation of adversarial alignment:", adv_loss_std)
