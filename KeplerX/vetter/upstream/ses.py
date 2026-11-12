
from fast_histogram import histogram1d
from .utils import *
from scipy.ndimage import median_filter

from scipy.fftpack import rfft, irfft
import pywt
import batman


def ocwt(x, max_level=14):
    level = min(max_level, np.log2(len(x))-1 )
    return np.array(pywt.swt(x, 'db2', trim_approx=True, norm=True, level=level) )[::-1]


    

def get_transit_signal(width, depth=100., limb_dark=[0.4012, 0.5318, -0.2411,0.0194], exp_time=0.020417, pad=65536, b=0., n_width=1.):

    if len(limb_dark) != 4:
        limb_dark = [0.4012, 0.5318, -0.2411, 0.0194]

    rp_rs = np.sqrt(depth/1.0e6)
    
    params = batman.TransitParams()
    params.a = 20.                        
    params.t0 = 0.                        
    params.per = np.pi*(width)/np.arcsin( ((1+rp_rs)**2. - b**2.)/params.a)
    params.inc = np.arccos(b/params.a)*(180./np.pi)                      
    params.ecc = 0.                       
    params.w = 90.                        
    params.limb_dark = "nonlinear"       
    params.u = limb_dark 
    
    n_points = n_width*width/exp_time
    n_pad = int(pad - 2*n_points)
    
    #t = np.linspace(-3.*width , 3.*width, int(2*np.ceil(n_points)) )

    t = np.concatenate([(-np.arange(exp_time, n_width*width, exp_time))[::-1], np.arange(0., n_width*width, exp_time)])
    
    params.rp =  rp_rs
    
    m = batman.TransitModel(params, t, exp_time=exp_time, fac=0.001, supersample_factor=5)
    lc = m.light_curve(params, )

    if len(lc)>pad:
        print(pad, len(lc), width )
        print('What the hell? Light curve is longer than padded light curve')

    ends = np.array_split(np.ones( pad - len(lc) ), 2)
    
    return np.concatenate( [ends[0], lc, ends[1] ])




def calc_var_stat(x, window_size, exp_time, method='mad',
                  slat_frac=0.2, n_mad_window=15_001):

    if method=='mean':
        window_points = 2*int(window_size/exp_time)
        sig2 = moving_average( x**2., n=window_points+1)

        return sig2
    
    elif method=='mad':
        
        window_points = 2*int(window_size/exp_time)

        if window_points<n_mad_window:
            sig = 1.4826 * running_median_insort(np.abs(x), window_points)
        else:
            dx  = window_points/n_mad_window
            indices = np.arange(len(x) )
            skip_indices = np.linspace(0,len(x), int(len(x)/dx) )

            x_skipped = np.interp(skip_indices, indices, x )
            
            #x_skipped = x[0::nskip] 
            sig_decimated = 1.4826*running_median_insort(np.abs(x_skipped), n_mad_window)
            
            #sig = np.repeat(sig_decimated, nskip)[:len(x)]
            sig =  np.interp(indices, skip_indices, sig_decimated )

        if (sig==0).any():
            nbad = np.sum(sig==0)
            print('WARNING! VAR STAT == 0 AT {}/{} CADENCES'.format(nbad,len(sig)))
            sig[np.where(sig==0)[0]] = np.nanmedian(sig)

        return sig**2.

    elif method=='gap_mean':
        window_points = 2*int(window_size/exp_time)

        slat_points = int(window_points/2)
        delta_slat = int((slat_frac)*window_points/2)

        delta_slat = int(delta_slat/2)*2
        slat_points = int(slat_points/2)*2
        
        sig2_left = moving_average( np.roll(x**2., -delta_slat), n=slat_points-1)
        sig2_right = moving_average(np.roll(x**2., delta_slat), n=slat_points-1)
        
        return 0.5*(sig2_left+sig2_right)



    
def get_whitening_coeffs(t, x, window_size,exp_time=0.020417,method='mad',
                  slat_frac=0.2, n_mad_window=51):


    for i,seg in seg_dates:

        if i<len(seg_dates)-1:
            x_seg = x[np.logical_and(t>=seg and t<seg_dates[i+1])]
        else:
            x_seg = x[np.logical_and(t>=seg and t<seg_dates[i+1])]

        x_seg_pad = pad_time_series()

    return 1.




def calculate_SES(flux_transform, signal_transform, window_size, var_calc='mad', texp=0.020417,):

    # Calculate the Doubly-Whitened Coefficients
    sig2 = [1./calc_var_stat(x, window_size*2**i, method=var_calc, exp_time=texp) for i,x in enumerate(flux_transform)]
    
    sig2 = np.array(sig2, )
    
    n_levels = len(flux_transform)
        
    N = np.zeros(len(sig2[0]) )
    D = np.zeros(len(sig2[0]) )

    levels = np.concatenate([np.arange(1,n_levels), [n_levels-1] ] )
    
    for i, l in enumerate(levels):
        
        N +=  2.** (-l) * convolve(sig2[i]*flux_transform[i], signal_transform[i,::-1], mode='same')
        D +=  2.** (-l) * convolve(sig2[i], signal_transform[i,::-1]**2., mode='same')    
    
    ses = N / np.sqrt(D)
    
    return ses, N, D, sig2







def calc_mes(fold_time, num, den, P, n_trans=2, texp=0.0204, norm=False,return_nans=False):

    nbins = int(P//texp)+1
    bin_range = (0, P)
    bin_edge = np.linspace(bin_range[0], bin_range[1], nbins)

    num_binned = histogram1d(fold_time, weights=num, bins=nbins, range=bin_range)
    den_binned = histogram1d(fold_time, weights=den, bins=nbins, range=bin_range)

    n_transits = histogram1d(fold_time, bins=nbins, range=bin_range)
    
    transit_cut = n_transits<n_trans

    den_binned[transit_cut]=1.
    num_binned[transit_cut]=0.
    
    mes = num_binned / np.sqrt(den_binned)

    if norm:
        mes -= np.nanmedian(mes[transit_cut])
        mes/=mad(mes[transit_cut])

    #mes[transit_cut] = 0.

    if return_nans:
        mes[n_transits==0]=np.nan    
    
    return bin_edge, mes



def calc_mes_fast(timefold_bins, num, den):

    numbinned = np.bincount(timefold_bins, weights=num) 
    denbinned = np.bincount(timefold_bins, weights=den) 
    
    mes = numbinned/np.sqrt(denbinned)

    return mes



def calc_mes_loop(timefold_bins, num, den):
        
    mes_all = [calc_mes_fast(timefold_bins, num[i], den[i]) for i in range(num.shape[0])]
    return np.array(mes_all)






