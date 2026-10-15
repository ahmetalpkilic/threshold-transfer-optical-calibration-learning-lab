"""Independent rank calibration and covariance-transfer control; synthetic data."""
from pathlib import Path
import json
import math
import numpy as np
from characterization import ar_sequence, fit_rho
import clock_model as c
import policy

SEED=20261015
ALPHA=.05

def rank_threshold(scores, alpha=ALPHA):
    """Strict exceedance of kth order statistic; never interpolate quantiles.

    Exchangeability gives P(new > kth) <= (n+1-k)/(n+1), marginally
    over calibration and new draws. Ties make strict rejection conservative.
    """
    scores=np.asarray(scores,float)
    if scores.ndim != 1 or len(scores)<1 or not np.all(np.isfinite(scores)):
        raise ValueError('Require a nonempty finite score vector')
    if not 0<alpha<1: raise ValueError('Require 0<alpha<1')
    k=math.ceil((len(scores)+1)*(1-alpha))
    if k>len(scores): raise ValueError('Too few calibration scores for finite threshold')
    return float(np.sort(scores)[k-1]),k


def null_scores(n, seed):
    """Each draw regenerates BOTH old (80/class) and pilot (40/class).

    This controls a new old+pilot pair, not all pilots conditional on a single
    deployed old estimate. These synthetic offline draws cost no acquisitions.
    """
    rng=np.random.default_rng(seed); scores=[]
    cov=c.covariances(4.)
    for _ in range(n):
        old=[];pilot=[]
        for g,rho in enumerate((0.,.4)):
            old.append(fit_rho(ar_sequence(rng.normal(size=(80,2)),rho,cov[g],np.zeros(2))))
            pilot.append(fit_rho(ar_sequence(rng.normal(size=(40,2)),rho,cov[g],np.zeros(2))))
        scores.append(float(np.max(np.abs(np.array(old)-pilot))))
    return scores


def nuisance_covariance():
    """Change variances AND within-pair correlation, retaining separability.

    A acts on centered two-clock errors only; latent clock means stay fixed.
    All current roles change together; old characterization keeps original C.
    """
    a=np.array([[1.5,.4],[-.2,.8]])
    return np.einsum('ij,njk,lk->nil',a,c.covariances(4.),a)


def experiment(n_train=999,trials=400,seed=SEED):
    # Distinct root children ensure threshold tuning cannot inspect test errors.
    train_seed,test_seed=np.random.SeedSequence(seed).spawn(2)
    scores=null_scores(n_train,train_seed)
    threshold,k=rank_threshold(scores)
    # Common random numbers pair the two covariance settings. Recreate the test
    # integer child seed before each policy.experiment call.
    child=lambda: int(test_seed.generate_state(1)[0])
    base=policy.experiment(trials,child(),threshold,levels=(.4,.9))
    changed=policy.experiment(trials,child(),threshold,levels=(.4,.9),current_cov=nuisance_covariance())
    rows=[]
    for label,result in [('base',base),('changed covariance',changed)]:
        for row in result['rows']: rows.append(dict(covariance=label,**row))
    differences=[max(abs(np.array(a['alarm_scores'])-b['alarm_scores'])) for a,b in zip(base['rows'],changed['rows'])]
    return dict(seed=seed,n_train=n_train,trials=trials,alpha=ALPHA,rank=k,threshold=threshold,
                marginal_false_alarm_bound=(n_train+1-k)/(n_train+1),training_scores=scores,
                covariance_base_s2=c.covariances(4.)[:2].tolist(),covariance_changed_s2=nuisance_covariance()[:2].tolist(),
                max_paired_score_difference=float(max(differences)),center_sd_px=base['center_sd_px'],
                sigma_px=base['sigma_px'],total_pairs=640,failed_fits=0,rows=rows)


def demo(out='outputs'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result=experiment(); out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    fig,axs=plt.subplots(1,3,figsize=(16,5.5),layout='constrained')
    rows=result['rows']; x=np.arange(4)
    labels=['Base\nno change','Base\nrho .4→.9','New C\nno change','New C\nrho .4→.9']
    axs[0].errorbar(x,[r['alarm_rate'] for r in rows],yerr=[2*r['alarm_mcse'] for r in rows],fmt='o',color='#255f85',capsize=4)
    axs[0].axhline(ALPHA,color='#333333',ls=':',label='5% null target')
    axs[0].set(title='Alarm rate on held-out trials',ylabel='Fraction ±2 conditional MCSE',ylim=(-.04,1.04));axs[0].legend()
    for name,color,marker in [('stale','#255f85','o'),('refresh','#c88322','s'),('conditional','#80658d','^'),('oracle','#555555','x')]:
        axs[1].plot(x,[r['methods'][name]['rmse_ms'] for r in rows],marker,ls='none',color=color,label=name)
    axs[1].set(title='Fresh optical-delay error',ylabel='RMSE (ms)',ylim=(0,None));axs[1].legend(fontsize=9)
    for j,(name,color,marker) in enumerate([('conditional minus stale','#255f85','o'),('conditional minus refresh','#c88322','s')]):
        axs[2].errorbar(x+(j-.5)*.15,[r['paired_risk'][name]['difference_ms2'] for r in rows],yerr=[2*r['paired_risk'][name]['mcse_ms2'] for r in rows],fmt=marker,color=color,capsize=3,label=name)
    axs[2].axhline(0,color='#333333',ls=':');axs[2].set(title='Paired measurement-risk difference',ylabel='MSE difference ±2 MCSE (ms²)');axs[2].legend(fontsize=8)
    for ax in axs:ax.set_xticks(x,labels);ax.grid(axis='y',alpha=.2)
    fig.suptitle(f'Independent threshold calibration: n={result["n_train"]}, h={result["threshold"]:.4f}; test n={result["trials"]}/scenario')
    fig.savefig(out/'experiment.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k not in ('training_scores','rows')},indent=2))
    for row in rows:print(row['covariance'],row['current_rho'],row['alarm_rate'],{m:round(s['rmse_ms'],4) for m,s in row['methods'].items()})

if __name__=='__main__':demo()
