"""Linear magnetisation backgrounds calibrated from an independent blank."""
import numpy as np
from scipy import stats


def estimate_from_blank(h,blank_m):
    h=np.asarray(h,float); m=np.asarray(blank_m,float)
    if len(h)<8 or np.ptp(h)<=0: raise ValueError('Independent blank needs >=8 field points')
    fit=stats.linregress(h,m)
    residual=m-fit.intercept-fit.slope*h
    z=(h-h.mean())/np.ptp(h)
    quadratic=stats.linregress(z*z,residual)
    return dict(chi=fit.slope,offset=fit.intercept,chi_se=fit.stderr,
        chi_interval_95=[fit.slope-stats.t.ppf(.975,len(h)-2)*fit.stderr,
                         fit.slope+stats.t.ppf(.975,len(h)-2)*fit.stderr],
        detected=bool(fit.pvalue<.05),linear_model_rejected=bool(quadratic.pvalue<.01),
        residual_rmse=float(np.sqrt(np.mean(residual**2))),n=len(h),
        interpretation='independent blank calibration, not phase identification')


def subtract(curves,calibration):
    if calibration['linear_model_rejected']:
        raise ValueError('Independent blank rejects a linear background model')
    return [dict(c,M=np.asarray(c['M'])-calibration['offset']-calibration['chi']*np.asarray(c['H'])) for c in curves]
