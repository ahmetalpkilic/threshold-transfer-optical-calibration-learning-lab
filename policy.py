"""Independent pilot alarms, acquisition costs, and optical-delay measurement."""
from pathlib import Path
import json
import numpy as np
import optics as o
import clock_model as c
import repeat_model as r
from characterization import ar_sequence, fit_rho

OLD_RHO = .4
LEVELS = (0., .4, .7, .9)
THRESHOLD = .25  # predetermined engineering threshold; not a significance level
SEED = 20261014
METHODS = ('stale', 'refresh', 'pilot_never', 'pilot_always', 'conditional', 'oracle')


def alarm(old, pilot, threshold=THRESHOLD):
    """Family alarm: either class changes by more than a fixed rho distance."""
    old, pilot = np.asarray(old), np.asarray(pilot)
    if old.shape != (2,) or pilot.shape != (2,) or not np.all(np.isfinite([old,pilot])):
        raise ValueError('Require two finite class estimates')
    if not np.isfinite(threshold) or threshold < 0: raise ValueError('Invalid threshold')
    return bool(np.max(np.abs(pilot-old)) > threshold)


def allocation(name, triggered=False):
    """Return pilot pairs, refresh pairs, repeats per event (old cost =160)."""
    return {'stale':(0,0,12), 'refresh':(0,80,10),
            'pilot_never':(80,0,10), 'pilot_always':(80,80,8),
            'conditional':(80,80,8) if triggered else (80,0,10)}[name]


def estimate_policy(repeats, old, refreshed, name, triggered=False):
    """Pilot values never become whitening estimates; evaluation truth excluded."""
    _, refresh_pairs, m = allocation(name,triggered)
    used = refreshed if refresh_pairs else old
    return r.estimate(repeats[:,:m],np.asarray(used)[r.LABELS])


def experiment(trials=300, seed=SEED, threshold=THRESHOLD, levels=LEVELS, current_cov=None):
    if not isinstance(trials,int) or trials < 2: raise ValueError('trials must be integer >=2')
    alarm([0,0],[0,0],threshold)  # validate before expensive computation
    image_rng, eval_rng, cal_rng, old_rng, pilot_rng, fresh_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(6)]
    cov=c.covariances(4.); means=np.column_stack((c.SYNC,c.RATE*c.SYNC+c.OFFSET))
    center=o.ORIGIN+o.VELOCITY*(c.RATE*(c.FRAME+c.DELAY)+c.OFFSET)
    centers=[o.fit_center(o.image(center)+image_rng.normal(0,o.READ_SD,33)) for _ in range(trials)]
    evaluation=eval_rng.normal(size=(trials,40,2)); calibration=cal_rng.normal(size=(trials,40,12,2))
    old_normal=old_rng.normal(size=(trials,2,80,2)); pilot_normal=pilot_rng.normal(size=(trials,2,40,2)); fresh_normal=fresh_rng.normal(size=(trials,2,40,2))
    old_fits=np.array([[fit_rho(ar_sequence(old_normal[k,g],[0.,OLD_RHO][g],cov[g],means[g])) for g in range(2)] for k in range(trials)])
    if current_cov is not None: cov=np.asarray(current_cov,float)
    rows=[]
    for rho in levels:
        actual=np.array([0.,rho]); errors={name:[] for name in METHODS}; alarms=[]; pilots=[]; refreshes=[]
        for k,u in enumerate(centers):
            # Separate data roles: alarm selection cannot inspect calibration or final errors.
            pilot=np.array([fit_rho(ar_sequence(pilot_normal[k,g],actual[g],cov[g],means[g])) for g in range(2)])
            triggered=alarm(old_fits[k],pilot,threshold)
            # Simulate independent refresh data for coupled always-refresh controls.
            # The conditional policy receives it only when triggered and pays for it then.
            fresh=np.array([fit_rho(ar_sequence(fresh_normal[k,g],actual[g],cov[g],means[g])) for g in range(2)])
            repeats=r.draw_repeats(calibration[k],cov,actual[r.LABELS],means)
            z=means+np.einsum('nij,nj->ni',np.linalg.cholesky(cov),evaluation[k])
            for name in METHODS:
                s=cov if name=='oracle' else estimate_policy(repeats,old_fits[k],fresh,name,triggered)
                fit=c.calibrate(*z.T,s)
                errors[name].append(1000*(c.delay_from_center(u,fit)-c.DELAY))
            alarms.append(triggered); pilots.append(pilot); refreshes.append(fresh)
        rate=float(np.mean(alarms)); paired={}
        for a,b in [('conditional','stale'),('conditional','refresh'),('conditional','pilot_never'),('conditional','pilot_always')]:
            d=np.square(errors[a])-np.square(errors[b])
            paired[a+' minus '+b]={'difference_ms2':float(d.mean()),'mcse_ms2':float(d.std(ddof=1)/np.sqrt(trials))}
        rows.append(dict(current_rho=rho,alarm_rate=rate,alarm_mcse=float(np.sqrt(rate*(1-rate)/trials)),
            false_alarm_rate=rate if rho==OLD_RHO else None,miss_rate=1-rate if rho!=OLD_RHO else None,
            conditional_mean_repeats=10-2*rate,
            alarm_scores=np.max(np.abs(np.asarray(pilots)-old_fits),axis=1).tolist(),
            methods={name:dict(rmse_ms=float(np.sqrt(np.mean(np.square(e)))),bias_ms=float(np.mean(e)),sd_ms=float(np.std(e,ddof=1))) for name,e in errors.items()},
            rho_estimates={name:dict(mean=np.mean(x,axis=0).tolist(),boundary_fraction=np.mean(np.abs(x)>=.95-1e-7,axis=0).tolist()) for name,x in [('old',old_fits),('pilot',pilots),('refresh',refreshes)]},paired_risk=paired))
    return dict(seed=seed,trials=trials,threshold=threshold,total_pairs=640,old_pairs=160,fresh_evaluation_pairs=40,
                sigma_px=o.SIGMA,center_sd_px=float(np.std(centers,ddof=1)),failed_fits=0,rows=rows)


def demo(out='outputs'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result=experiment();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(16,4.8),layout='constrained')
    rows=result['rows']
    axes[0].errorbar(LEVELS,[x['alarm_rate'] for x in rows],yerr=[2*x['alarm_mcse'] for x in rows],fmt='o-',capsize=3)
    axes[0].axvline(OLD_RHO,color='k',ls=':',label='No change');axes[0].set(title='Pilot alarm rate ±2 MCSE',ylabel='Fraction of trials',ylim=(-.05,1.05))
    for name in ('stale','refresh','conditional','oracle'):
        axes[1].plot(LEVELS,[x['methods'][name]['rmse_ms'] for x in rows],'o-',label=name)
    axes[1].set(title='Fresh optical-delay measurement',ylabel='Delay RMSE (ms)')
    for name in ('conditional minus stale','conditional minus refresh'):
        axes[2].errorbar(LEVELS,[x['paired_risk'][name]['difference_ms2'] for x in rows],yerr=[2*x['paired_risk'][name]['mcse_ms2'] for x in rows],fmt='o-',capsize=3,label=name)
    axes[2].axhline(0,color='k',ls=':');axes[2].set(title='Paired risk difference ±2 MCSE',ylabel='MSE difference (ms²)')
    for ax in axes:ax.set(xlabel='Current odd-class rho (old = 0.4)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(out/'experiment.png',dpi=160);plt.close(fig)
    print(json.dumps(result,indent=2))

if __name__=='__main__':demo()
