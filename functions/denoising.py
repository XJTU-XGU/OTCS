import torch
import tqdm


def compute_alpha(beta, t, r=1):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta/r).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def ddpm_steps_conditioned_on_features(x, seq, model,feat_source, b):
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    xs = [x]
    x0_preds = []
    betas = b
    for i, j in tqdm.tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(betas, t.long())
        atm1 = compute_alpha(betas, next_t.long())
        beta_t = 1 - at/atm1
        x = xs[-1].to('cuda')
        with torch.no_grad():
            output = model(x, t.float(),feat_source)
            output, _ = torch.split(output, 3, dim=1)
        e = output

        x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
        x0_from_e = torch.clamp(x0_from_e, -1, 1)
        x0_preds.append(x0_from_e.to('cpu'))
        mean_eps = (
            (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
        ) / (1.0 - at)

        mean = mean_eps
        noise = torch.randn_like(x)
        mask = 1 - (t == 0).float()
        mask = mask.view(-1, 1, 1, 1)
        logvar = beta_t.log()
        sample = mean + mask * torch.exp(0.5 * logvar) * noise
        xs.append(sample.to('cpu'))
    return xs[-1]


def ddpm_steps_conditioned_on_images(x, seq, model,y, b, source_init=True,init_time=0.2):

    if source_init:
        expend_time = 1/init_time
        T = seq[int(len(seq) * init_time)]
        T_ = (torch.ones(len(x)) * T).to(x.device)
        a = compute_alpha(b,T_.long())
        e = x
        x = y * a.sqrt() + e * (1.0 - a).sqrt()
        seq = range(0,int(expend_time*T))
        b = torch.linspace(b[0],b[T],int(expend_time*T)).cuda()
    else:
        expend_time = 1

    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    xs = [x]
    x0_preds = []
    betas = b
    for i, j in tqdm.tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(betas, t.long(),expend_time)
        atm1 = compute_alpha(betas, next_t.long(),expend_time)
        beta_t = 1 - at/atm1
        x = xs[-1].to('cuda')
        with torch.no_grad():
            output = model(x, t.float()/expend_time,y)
        e = output

        x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
        x0_from_e = torch.clamp(x0_from_e, -1, 1)
        x0_preds.append(x0_from_e.to('cpu'))
        mean_eps = (
            (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
        ) / (1.0 - at)

        mean = mean_eps
        noise = torch.randn_like(x)
        mask = 1 - (t == 0).float()
        mask = mask.view(-1, 1, 1, 1)
        logvar = beta_t.log()
        sample = mean + mask * torch.exp(0.5 * logvar) * noise
        xs.append(sample.to('cpu'))
    return xs[-1]


