import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as pat
from scipy.interpolate import RectBivariateSpline
from scipy.integrate import quad
import corner
pi = np.pi


def planet_P(a,Ms):
    return (((a**3.0)/Ms)**0.5)*365.25 # Gives you the planet's orbital period in days

def planet_Tirrad(Ts,Rs,a):
    return Ts*(Rs/(a*215.03))**0.5  # Temp. in Kelvin; converts AU into Solar radii

def planet_Tdaynight(vep,AB,T0):
    Td = T0*((1.0-AB)**0.25)*(((2.0/3.0)-(5.0*vep/12.0))**0.25)
    Tn = T0*((1.0-AB)**0.25)*((vep/4.0)**0.25)
    return Td,Tn

def LC_offdeg(vep):
    if vep == 0:  # Don't worry about vep == 1 since amplitude should be very low anyway
        odeg = 0
    else:
        odeg = np.sum(np.array([114.40485701,-121.87513627,-11.3030972,94.83887575,2.29698976])*
                      np.array([vep**4.0,vep**3.0,vep**2.0,vep,1.0]))  # Derived from Cowan & Agol, 2011a
    return odeg

# Wave and bndwid should be given in microns
def planck(wave,bndwid,Tbody):
    wave_m,bndwid_m = wave*(1e-6),bndwid*(1e-6)
    lightspeed = 2.99792458e8
    hconst = 6.62606957e-34
    boltz = 1.3806488e-23
    plnk_curv = lambda w: (pi*(2.0*hconst*(lightspeed**2.0)/(w**5.0))/
                           (np.exp(hconst*lightspeed/(w*boltz*Tbody)) - 1.0))
    flux = quad(plnk_curv,wave_m-0.5*bndwid_m,wave_m+0.5*bndwid_m)[0]
    return flux

## The time formulas here were helped by Paul Anthony Wilson's website "Exoplanet Transit Method"
def LC_depths_durations(wave,bndwid,Rs,Ts,a,porb_dys,b,Rp,Td,Tn,Ag,odeg):
    ajup,rsjup = a*215.03*9.955,Rs*9.955  # Converts both to Jupiter radii
    porb_sec = porb_dys*24*3600  # Porb in sec

    dtr_max = (Rp/rsjup)**2.0
    
    tr_modify = np.cos(b*pi/2.0)**0.25
    dtr_nom = (0.7 + 0.3*tr_modify)*dtr_max  # Simulates non-zero impact parameters (up to 30% decrease in flux)
    pwr_tr = 2.5*tr_modify + 1.5
    leng_tr = porb_sec*np.arcsin((((rsjup + Rp)**2.0 - (b*rsjup)**2.0)**0.5)/ajup)/pi
    
    # Simulates eclipse depth being a bit larger than transit depth suggests (email to Nick 2/10/17)
    dtr_eff = 0.5*((2.0 - b)*dtr_max + b*dtr_nom)
    if wave == 'Bolo':
        dec_nom = dtr_eff*(Td/Ts)**4.0 + Ag*(Rp/ajup)**2.0
    else:
        beta_S,beta_Pd = planck(wave,bndwid,Ts),planck(wave,bndwid,Td)
        dec_nom = dtr_eff*(beta_Pd/beta_S) + Ag*(Rp/ajup)**2.0
    critical_b = (rsjup - Rp)/rsjup
    if b <= critical_b:
        dlt_ecl = dec_nom
        leng_ecl = porb_sec*np.arcsin((((rsjup - Rp)**2.0 - (b*rsjup)**2.0)**0.5)/ajup)/pi  # Note first sign change
        dlt_tr = dtr_nom
    else:
        if b == 1:
            dlt_ecl = 0.5*dec_nom  # Half of disk covered
            dlt_tr = 0.5*dtr_nom
        else:
            bfrac = ((b - critical_b)/(1.0 - critical_b))  # Calculating area of planet still covered, with next line
            ecl_modify = np.arctan((2.0*bfrac - bfrac**2.0)/(1.0 - bfrac)) - (2.0*bfrac - bfrac**2.0)*(1.0 - bfrac)
            dlt_ecl = (1.0 - ecl_modify/(2.0*pi))*dec_nom
            dlt_tr = (1.0 - ecl_modify/(2.0*pi))*dtr_nom
        leng_ecl = 0
    leng_ineg = 0.5*(leng_tr - leng_ecl)  # Duration of ingress or egress, individually
    
    if wave == 'Bolo':
        dlt_var = (dec_nom - dtr_eff*(Tn/Ts)**4.0)*(1.0/np.cos(odeg*pi/180.0))
    else:
        beta_Pn = planck(wave,bndwid,Tn)
        dlt_var = (dec_nom - dtr_eff*(beta_Pn/beta_S))*(1.0/np.cos(odeg*pi/180.0))
    
    dlt_flx = dec_nom + 0.5*dlt_var*np.cos(pi + (odeg*pi/180.0))  # Flux offset; puts bottom of dec_nom at 1.0
    return dlt_tr,pwr_tr,leng_tr,dlt_ecl,leng_ecl,leng_ineg,dlt_var,dlt_flx

#####

def pregen_Nvals(n_per_hr,orbs_seen,Porb):
    n_per_orb = int(n_per_hr*Porb/3600.0)  # Assuming Porb is in seconds
    n_data = int(orbs_seen*n_per_orb)
    return n_per_orb,n_data


def tr_ecl_maker(flux,t_dummy,dip,Porb,kind,style,exact_flxs):
    hwid,ineg,depth = dip  # 'ineg' is a power for round dips, time duration for flat dips
    if kind == 'T':
        mid_t = 0
    elif kind == 'E':
        mid_t = 0.5*Porb
    occult = np.logical_and((mid_t - hwid) <= t_dummy,
                            t_dummy <= (mid_t + hwid))
    if style == 'Round':
        scaling = -depth/(hwid**ineg)
        f_change = scaling*np.absolute(t_dummy[occult] - mid_t)**ineg
        f_change += depth  # This all creates an 'upside-down' curve that you subtract from the flux (next line)
        flux[occult] -= f_change
    elif style == 'Flat':
        tin,teg = mid_t - hwid - ineg,mid_t + hwid + ineg
        fin,fcen,feg = exact_flxs
        
        fcen_depress = fcen - depth
        flux[occult] = fcen_depress
        
        ingress = np.logical_xor(np.logical_and(tin <= t_dummy,t_dummy <= (mid_t + hwid)),
                                 occult)  # i.e. From start of ingress but NOT in-eclipse
        flux[ingress] = fcen_depress + (fin - fcen_depress)*((mid_t - hwid) - t_dummy[ingress])/ineg  # Simple slope 1
        
        egress = np.logical_xor(np.logical_and((mid_t - hwid) <= t_dummy,t_dummy <= teg),
                                occult)  # i.e. Til end of egress but NOT in-eclipse
        flux[egress] = fcen_depress + (feg - fcen_depress)*(t_dummy[egress] - (mid_t + hwid))/ineg  # Simple slope 2
    return flux

def perf_astro_model(t_full,t_dummy,astro,trans,ecl):
    amp,Porb,phi_off,dlt_flx,tr_cent = astro
    ehwid,ineg,edepth = ecl  # Don't use edepth at moment but just in case for future
    time_range = t_full[-1] - t_full[0]  # Redundant since t_full[0] should be 0, but just in case...
    
    #### First, use the dummy time vector to build a 'standardized' orbit
    ## astro[]: 0 amplitude, 1 orbital period, 2 phase offset, 3 time of transit center (first orbit)
    ## trans[]: 0 half transit duration, 1 power (i.e. exponent) describing curve, 2 depth
    ## ecl[]: 0 half in-eclipse duration, 1 duration of ingress/egress, 2 depth; time fixed half-orbit from transit
    a_mdl = amp*(-np.cos((t_dummy*2.0*pi/Porb) + phi_off)) + 1.0
    ec_t = 0.5*Porb  # Since transit occurs at 0 to begin with
    edip_tms = np.array([ec_t - ehwid - ineg,ec_t,ec_t + ehwid + ineg])  # Start, center, end eclipse+gresses times
    ef_vls = amp*(-np.cos((edip_tms*2.0*pi/Porb) + phi_off)) + 1.0  # Phase curve flux at above times
    a_mdl = tr_ecl_maker(a_mdl,t_dummy,trans,Porb,'T','Round','N/A')  # flat if wanted (would need 'tf_vls')
    a_mdl = tr_ecl_maker(a_mdl,t_dummy,ecl,Porb,'E','Flat',ef_vls)  # round if wanted
    a_mdl += dlt_flx  # Puts (nominal) bottom of eclipse at 1.0 (if perfect; i.e simulates dividing light curve by stellar flux)
    
    ## Next, roll the 'standardized' orbit, tile as needed, and get the length of data that matters
    a_mdl = np.roll(a_mdl,int(len(t_dummy)*(tr_cent/Porb - 0.25)))  # IMPORTANT: t_dummy began at -0.25 orbital phase
    a_mdl = np.tile(a_mdl,np.ceil(time_range/Porb))
    a_mdl = a_mdl[:len(t_full)]
    return a_mdl


def poly_detect_model(x_o,y_o,dC_A):  # dC_A = detector Coefficient Array
    ## Wrapping centroid values into central pixel (decent guess if all pixels have roughly same sensitivity map)
    wrapxo = ((x_o-0.5) % 1.0) + 14.5
    wrapyo = ((y_o-0.5) % 1.0) + 14.5
    
    d_mdl = np.polynomial.polynomial.polyval2d(wrapxo-15.0,wrapyo-15.0,dC_A)
    return d_mdl


def perf_detect_model(x_o,y_o):
    ## 'Quenching' the sensitivity around the pixel
    xbndry = np.poly1d([14.49,15.51],True)
    ybndry = np.poly1d([14.49,15.51],True)
    
    ## Wrapping centroid values into central pixel
    wrapxo = ((x_o-0.5) % 1.0) + 14.5
    wrapyo = ((y_o-0.5) % 1.0) + 14.5
    
    d_mdl = xbndry(wrapxo)*ybndry(wrapyo)
    
    ## Make only positive values (scale things within 'Dm_plus_noise')
    d_mdl = np.absolute(d_mdl)
    return d_mdl


def preNoi_gridding(x_o,y_o,pnt_per_px):
    lw_xpxcntr,hi_xpxcntr = np.round(np.amin(x_o)),np.round(np.amax(x_o))
    lw_ypxcntr,hi_ypxcntr = np.round(np.amin(y_o)),np.round(np.amax(y_o))
    xtotpx,ytotpx = hi_xpxcntr - lw_xpxcntr + 1,hi_ypxcntr - lw_ypxcntr + 1
    nx,ny = int(pnt_per_px*xtotpx),int(pnt_per_px*ytotpx)
    xgrd,ygrd = np.linspace(lw_xpxcntr-0.5,hi_xpxcntr+0.5,nx),np.linspace(lw_ypxcntr-0.5,hi_ypxcntr+0.5,ny)
    grdNoi = np.random.randn(ny,nx)
    return xgrd,ygrd,grdNoi


def Dm_plus_noise(x_o,y_o,xgrd,ygrd,grdNoi,levNoi,amp_want,i_t,which,viz_V):
    ## Windowing: only check data in middle 'w' % of 'which' pointing you have, for scaling stuff
    w,nd = 0.8,i_t[which+1] - i_t[which]
    
    d_mdl = perf_detect_model(x_o,y_o)
    ## Scaling surface to [0,1] under relevant centroids, as a standard (can change if you want)
    shift1 = np.amin(d_mdl[int(0.5*(1-w)*nd):int(0.5*(1+w)*nd)])
    d_mdl -= shift1
    scale1 = np.amax(d_mdl[int(0.5*(1-w)*nd):int(0.5*(1+w)*nd)])
    d_mdl /= scale1
    
    ## Spliny interpolated noise; np.transpose puts input [y,x] into [x,y] needed here
    ## Choose noise level relative to the above [0,1] standard
    noisy_spline = RectBivariateSpline(xgrd,ygrd,np.transpose(grdNoi),kx=2,ky=2)
    noize_mdl = levNoi*noisy_spline(x_o,y_o,grid=False)
    
    ## Adding D and noise, rescaling for relevant centroids as needed, and shifting median to 1
    nominal_mdl = d_mdl + noize_mdl
    shift2 = np.amin(nominal_mdl[int(0.5*(1-w)*nd):int(0.5*(1+w)*nd)])
    nominal_mdl -= shift2
    scale2 = amp_want/np.amax(nominal_mdl[int(0.5*(1-w)*nd):int(0.5*(1+w)*nd)])
    final_mdl = scale2*nominal_mdl
    shift3 = 1.0 - np.median(final_mdl[int(0.5*(1-w)*nd):int(0.5*(1+w)*nd)])
    final_mdl += shift3
    
    ## Want to make a map for visualing?
    if viz_V == True:
        nx,ny = 5*len(xgrd),5*len(ygrd)
        VX,VY = np.meshgrid(np.linspace(xgrd[0],xgrd[-1],nx),
                            np.linspace(ygrd[0],ygrd[-1],ny))
        VX_rav,VY_rav = np.ravel(VX),np.ravel(VY)  # Flaten out to vectors
        vd_mod = perf_detect_model(VX_rav,VY_rav)
        # Repeat exactly all the steps you did to the actual D
        vd_mod -= shift1
        vd_mod /= scale1
        vn_mod = levNoi*noisy_spline(VX_rav,VY_rav,grid=False)
        vnom_mod = vd_mod + vn_mod
        vnom_mod -= shift2
        vfin_mod = scale2*vnom_mod
        vfin_mod += shift3
        v_sens = np.reshape(vfin_mod,(ny,nx))  # Un-ravel the map back into an array
        return final_mdl,v_sens,VX,VY
    
    return final_mdl

## Full generative model
def real_gen_model(xgrd,ygrd,grdNoi,levNoi,amp_want,i_t,which,viz_V,
                      n_per_orb,n_data,orbs_seen,astro,trans,ecl,x_o,y_o,mu_n,sig_n):
    Porb = astro[1]
    t_full = np.linspace(0,orbs_seen*Porb,n_data)
    t_dummy = np.linspace(-0.25*Porb,0.75*Porb,n_per_orb)  # IMPORTANT CONVENTION HREE- LEAVE ALONE!!
    a_mdl = perf_astro_model(t_full,t_dummy,astro,trans,ecl)
    if viz_V == True:
        d_mdl,v_sens,VX,VY = Dm_plus_noise(x_o,y_o,xgrd,ygrd,grdNoi,levNoi,amp_want,i_t,which,viz_V)
    else:
        d_mdl = Dm_plus_noise(x_o,y_o,xgrd,ygrd,grdNoi,levNoi,amp_want,i_t,which,viz_V)
    y = a_mdl*d_mdl
    y_data = y + (sig_n*np.random.randn(n_data) + mu_n)
    if viz_V == True:
        return t_full,a_mdl,d_mdl,y,y_data,v_sens,VX,VY
    else:
        return t_full,a_mdl,d_mdl,y,y_data

def brown_signal(bkey,rat_BD,y,yd,a_mdl,d_mdl,muf,sigf,n_data):
    if bkey == True:
        B_noi,ig1 = reg_brown_mot(rat_BD*(np.amax(d_mdl) - np.amin(d_mdl)),rat_BD,n_data)
        B_noi += 1.0  # +1.0 so mean is around unity (because of gen. model)
        ynew = a_mdl*d_mdl*B_noi
        ydnew = ynew + sigf*np.random.randn(n_data) + muf
        return ynew,ydnew,B_noi
    else:
        B_noi = np.ones(n_data)
        return y,yd,B_noi

#####

### Definitions for making more realistic centroids
def projection_axes(thetaj,thetaw,thetasd,thetald):
    return (np.cos(thetaj),np.sin(thetaj),
            np.cos(thetaw),np.sin(thetaw),
            np.cos(thetasd),np.sin(thetasd),
            np.cos(thetald),np.sin(thetald))

def reg_brown_mot(Abmx,Abmy,n_data):
    bmx_steps = 2.0*np.random.random(n_data) - 1.0
    bmx_steps[0] = 0  # To keep first step at selected x-position
    time_bmx = np.cumsum(bmx_steps)
    time_bmx = Abmx*time_bmx/np.amax(np.absolute(time_bmx))  # Rescale for maximum excursion
    
    bmy_steps = 2.0*np.random.random(n_data) - 1.0
    bmy_steps[0] = 0  # Ditto for y-position
    time_bmy = np.cumsum(bmy_steps)
    time_bmy = Abmy*time_bmy/np.amax(np.absolute(time_bmy))
    return time_bmx,time_bmy

def wobble_amp_per_time(Aw,Pw,DAwmax,DPwmax,n_data):
    amp_steps = 2.0*np.random.random(n_data) - 1.0
    amp_steps[0] = 0  # To keep first step at selected amp
    time_amps = np.cumsum(amp_steps)
    time_amps = DAwmax*time_amps/np.amax(np.absolute(time_amps))  # Rescale for maximum excursion
    time_amps += Aw
    
    per_steps = 2.0*np.random.random(n_data) - 1.0
    per_steps[0] = 0  # Ditto for period
    time_pers = np.cumsum(per_steps)
    time_pers = DPwmax*time_pers/np.amax(np.absolute(time_pers))
    time_pers += Pw
    return time_amps,time_pers

def telescope_pointing(J_full,W_full,Shtd_full,Ltd_full,t,t0,n_data,cont):  # Jitter, Wobble, Short-, Long-Term Drift
    # Unpacking Variables- see Ingalls+2016 Appendix A for details
    Aj,Pj,phij,thetaj,Abmx,Abmy = J_full
    Aw,Pw,phiw,Sw,DAwmax,DPwmax,thetaw = W_full
    Asd,Psd,phisd,tausd,thetasd = Shtd_full
    Ald,thetald = Ltd_full
#     tot_sec = t[-1] - t[0]
    
    # Projection Axes
    c_thj,s_thj,c_thw,s_thw,c_thsd,s_thsd,c_thld,s_thld = projection_axes(thetaj,thetaw,thetasd,thetald)

    ### Jitter
    Jit_fun = Aj*np.sin((2.0*pi*(t-t0)/Pj) + phij)
#     FBM_noise_x,FBM_noise_y = frac_brown_mot(Afbm,Beta,n_data,tot_sec)
    RBM_noise_x,RBM_noise_y = reg_brown_mot(Abmx,Abmy,n_data)
    Jit_x = Jit_fun*c_thj + RBM_noise_x
    Jit_y = Jit_fun*s_thj + RBM_noise_y
    ###
    
    ### Wobble
    Awt,Pwt = wobble_amp_per_time(Aw,Pw,DAwmax,DPwmax,n_data)
    small_q = ((t-t0)/Pwt) + (phiw/(2.0*pi))
    wob_Low = np.logical_and(0 <= small_q,small_q < Sw)
    wob_Mid = np.logical_and(Sw <= small_q,small_q < (1-Sw))
    wob_Hig = np.logical_and((1-Sw) <= small_q,small_q < 1)
    phiskt = np.zeros(n_data)
    
    phiskt[wob_Low] = pi*((1.0/(2.0*Sw)) - 2.0)*small_q[wob_Low]
    phiskt[wob_Mid] = pi*(((small_q[wob_Mid] - Sw)/(1.0 - 2.0*Sw)) - 2.0*small_q[wob_Mid] + 0.5)
    phiskt[wob_Hig] = pi*((1.0/(2.0*Sw)) - 2.0)*(small_q[wob_Hig] - 1.0)
    
    Wob_fun = Awt*np.sin(2.0*pi*small_q + phiskt)
    Wob_x = Wob_fun*c_thw
    Wob_y = Wob_fun*s_thw
    ###
    
    ### Short-Term Drift
    ShTD_fun = (Asd/np.sin(phisd))*np.sin((2.0*pi*(t-t0)/Psd) + phisd)*np.exp(-(t-t0)/tausd)
    ShTD_x = ShTD_fun*c_thsd
    ShTD_y = ShTD_fun*s_thsd
#     ShTD_x = 0
#     ShTD_y = 0
    ###
    
    ### Long-Term Drift
    LTD_fun = Ald*(t-t0)
    LTD_x = LTD_fun*c_thld
    LTD_y = LTD_fun*s_thld
    ###
    
    ### Starting Centroid; either random point or nearby the end of the last pointing (will assume len-2 vector)
    if np.all(cont) == False:
        initial_x = 1.0*np.random.random() + 14.5  # 15.0
        initial_y = 1.0*np.random.random() + 14.5  # 15.0
    else:
        initial_x = cont[0] + (0.1*np.random.random() - 0.2)
        initial_y = cont[1] + (0.1*np.random.random() - 0.2)
    ###
    return initial_x+Jit_x+Wob_x+ShTD_x+LTD_x,initial_y+Jit_y+Wob_y+ShTD_y+LTD_y

### Parameters for generating centroids- see Ingalls+2016 Appendix A for details
def point_params():
    J_vals = np.array([0.04,60,
                       2.0*pi*np.random.random(),
                       -45.0*pi/180.0,
                       0.15*np.random.random() + 0.0375,
                       0.15*np.random.random() + 0.0375])
#                         0,
#                         0])
#                         0.2*np.random.random() + 0.05,
#                         0.2*np.random.random() + 0.05])  # Reg. B.M., not Frac. B.M. anymore

    W_vals = np.array([0.016*np.random.random() + 0.018,
                       1600*np.random.random() + 1200,2.0*np.random.random() - 1.0,
                       0.3*np.random.random() + 0.1,
                       0.01,10.0,
                       (-35.0*pi/180.0)*np.random.random() - (45.0*pi/180.0)])

    SD_vals = np.array([2.0*np.random.random() - 1.0,
                        395.6,7.0*pi/4.0,
                        np.absolute(1800*np.random.randn()),
                        5.0*pi/9.0])

    LD_vals = np.array([(0.0125/3600)*np.random.random(),
                        (-40.0*pi/180.0)*np.random.random() - (55.0*pi/180.0)])
    
    return J_vals,W_vals,SD_vals,LD_vals

#####

def centroiding(tt,repnts):
    tm_inds = np.around(repnts*len(tt)).astype(int)  # Time indices (return later)
    slct_pnt = np.random.randint(len(tm_inds)-1)  # For scaling later in generative model
    tweak = np.floor(np.indices((len(tm_inds),))[0]/(len(tm_inds)-1)).astype(int)  # Tweak LAST index to end of vector
    t_reset = tt[tm_inds - tweak]  # Sets initial time(s) in telescope_pointing

    xo,yo = np.zeros(tt.shape),np.zeros(tt.shape)  # Pre-sizing vectors for centroids
    pnt_continuity = False  # Randomize 1st pointing on center pixel; repoints stay ~nearby~ where last pointing ended
    for i in np.linspace(0,len(t_reset)-2,len(t_reset)-1).astype(int):  # i.e. for each repoint
        n_beg,n_end = tm_inds[i],tm_inds[i+1]
        n_block = n_end - n_beg
        J_vals,W_vals,SD_vals,LD_vals = point_params()  # Pick a pointing model
        if i > 0:
            pnt_continuity = np.array([xo[n_beg-1],xo[[n_beg-1]]])
        (xo[n_beg:n_end],
         yo[n_beg:n_end]) = telescope_pointing(J_vals,W_vals,SD_vals,LD_vals,tt[n_beg:n_end],t_reset[i],n_block,pnt_continuity)
    return xo,yo,tm_inds,slct_pnt,J_vals,W_vals,SD_vals,LD_vals

def mvG_noise(sx,sy,rho,n_data):  # Multivariate Gaussian Noise
    mu = np.array([0.0,0.0])
    cov = np.array([[sx**2.0,rho*sx*sy],[rho*sx*sy,sy**2.0]])
    D = np.random.multivariate_normal(mu,cov,n_data)
    dx,dy = D[:,0],D[:,1]
    return dx,dy

def centroid_jiggle(jigg,xo,yo,n_data,frac,r_max):
    if jigg == True:
        sx = frac*(np.amax(xo) - np.amin(xo))
        sy = frac*(np.amax(yo) - np.amin(yo))
        rho = r_max*(2.0*np.random.random() - 1.0)  # Takes values from -1.0 to 1.0, though probably don't use the extremes
        cov = np.array([sx,sy,rho])
        # Jiggling the true xo and yo, with covariance
        dx,dy = mvG_noise(sx,sy,rho,n_data)  
        xv = xo + dx
        yv = yo + dy
        return xv,yv,cov
    else:
        cov = np.array([0,0,0])
        return xo,yo,cov

#####

### BLISS-related definitions
def lh_axes_binning(x_o,y_o,b_n,n_data):
    plt.figure(figsize=(6,6))
    xy_C,x_edg,y_edg,viz = plt.hist2d(x_o,y_o,b_n,cmap=cm.Purples);
    plt.title('Data per Knot',size=24)
    plt.xlabel('$x_{0}$',size=24);
    plt.ylabel('$y_{0}$',size=24);
    plt.colorbar(viz)
    x_k = x_edg[1:] - 0.5*(x_edg[1:] - x_edg[0:-1])
    y_k = y_edg[1:] - 0.5*(y_edg[1:] - y_edg[0:-1])
    x_Knt,y_Knt = np.meshgrid(x_k,y_k)
    
    l_b_x,h_b_x = lh_knot_ass(x_k,x_o,b_n,n_data)
    l_b_y,h_b_y = lh_knot_ass(y_k,y_o,b_n,n_data)

    xy_C[xy_C == 0] = 0.1  # Avoid division by zero errors
    xy_C = np.transpose(xy_C)  # Because you want [y,x] and not [x,y] ordering

    plt.tight_layout()
    plt.show()
    return l_b_x,h_b_x,l_b_y,h_b_y,x_k,y_k,xy_C,x_edg,y_edg,x_Knt,y_Knt  # Edges and Knot mesh included


def lh_knot_ass(xy_k,xy_o,b_n,n_data):
    bad_l_xy = (xy_o < xy_k[0])  # pre-finding points "outside" the knots
    bad_h_xy = (xy_o > xy_k[-1]) 
    
    mid_xy_cln = np.transpose(np.tile(xy_k,(n_data,1)))
    diff_xy_cln = xy_o - mid_xy_cln
    diff_xy_cln[diff_xy_cln < 0] = (xy_k[-1] - xy_k[0])
    l_b_xy = np.argmin(diff_xy_cln**2.0,axis=0)
    
    diff_xy_cln = mid_xy_cln - xy_o
    diff_xy_cln[diff_xy_cln < 0] = (xy_k[-1] - xy_k[0])
    h_b_xy = np.argmin(diff_xy_cln**2.0,axis=0)
    
    l_b_xy[l_b_xy == b_n-1] = b_n-2  # tuning l_b_xy upper bound and vice versa
    h_b_xy[h_b_xy == 0] = 1
    h_b_xy[h_b_xy == l_b_xy] += 1  # Avoiding same bin reference (PROBLEMS?)
    
    l_b_xy[bad_l_xy] = 0  # manually extrapolating points "outside" the knots
    h_b_xy[bad_l_xy] = 1
    l_b_xy[bad_h_xy] = b_n-2
    h_b_xy[bad_h_xy] = b_n-1
    
    return l_b_xy,h_b_xy


def which_NNI(keynni,xy_Ct,l_b_x,h_b_x,l_b_y,h_b_y):
    bad_left = np.logical_or(xy_Ct[l_b_y,l_b_x] == 0.1,xy_Ct[l_b_y,h_b_x] == 0.1)
    bad_right = np.logical_or(xy_Ct[h_b_y,l_b_x] == 0.1,xy_Ct[h_b_y,h_b_x] == 0.1)
    nni_mask = np.logical_or(bad_left,bad_right)
    if keynni == True:
        nni_mask = np.ones(nni_mask.shape).astype(bool)
    return nni_mask,np.logical_not(nni_mask)


def lh_bin_to_knot(x_k,y_k,l_b_x,h_b_x,l_b_y,h_b_y):
    l_x_K = x_k[l_b_x]  # Index arrays are awesome!
    h_x_K = x_k[h_b_x]
    l_y_K = y_k[l_b_y]
    h_y_K = y_k[h_b_y]
    return l_x_K,h_x_K,l_y_K,h_y_K


def bound_knot(x_o,y_o,l_b_x,h_b_x,l_b_y,h_b_y,l_x_K,h_x_K,l_y_K,h_y_K,n_data):
    left = (x_o - l_x_K <= h_x_K - x_o)
    right = np.logical_not(left)
    bottom = (y_o - l_y_K <= h_y_K - y_o)
    top = np.logical_not(bottom)
    
    xo_B_i,yo_B_i = np.zeros(n_data),np.zeros(n_data)
    xo_B_i[left] = l_b_x[left]
    xo_B_i[right] = h_b_x[right]
    yo_B_i[bottom] = l_b_y[bottom]
    yo_B_i[top] = h_b_y[top]
    
    return xo_B_i.astype(int),yo_B_i.astype(int)


def bliss_dist(x_o,y_o,l_x_K,h_x_K,l_y_K,h_y_K):
    LLd = (h_x_K - x_o)*(h_y_K - y_o)
    LRd = (x_o - l_x_K)*(h_y_K - y_o)
    ULd = (h_x_K - x_o)*(y_o - l_y_K)
    URd = (x_o - l_x_K)*(y_o - l_y_K)
    return LLd,LRd,ULd,URd

#####

# ~Fast~ 'B'-type flux summing using 'np.bincount' (learned while dealing with slow covariance functions)
# Note use of 'minlength': you'd often get output that's too short otherwise! (i.e. not EVERY possible knot)
def map_flux_avgQuick(obs_data,a_model,dxyi_lin,n_k,boundkt):
    dxy_map = ((np.bincount(dxyi_lin,weights=obs_data/a_model,minlength=(n_k*n_k)).reshape((n_k,n_k)))/boundkt)  # Avg flux at each data knot
    return dxy_map  # Using [y,x] for consistency

# 'J'-type map making
def map_flux_jumped(dxy_2d,mask_bkt,theta_K):
    dxy_2d[mask_bkt] = theta_K  # Using "tmask_goodBKT" for consistency
    return dxy_2d

# This is the good version :-)
def bliss_meth(n_data,b_flux,dxy_map,xo,yo,dx,dy,xi,yi,
               lbx,hbx,lby,hby,ll,lr,ul,ur,bls,nni):
    LL = dxy_map[lby,lbx]*ll  # Using [y,x] for consistency
    LR = dxy_map[lby,hbx]*lr
    UL = dxy_map[hby,lbx]*ul
    UR = dxy_map[hby,hbx]*ur
    b_flux[bls] = (LL[bls] + LR[bls] + UL[bls] + UR[bls])/(dx*dy)  # BLISS points
    b_flux[nni] = dxy_map[yi[nni],xi[nni]]  # Nearest Neighbor points
    return b_flux

#####

def star_T_color(Ts):
    Tval = np.array([3050,4450,5600,6750,8750,20000,30000])  # Wikipedia temperature groups
    rgb = np.array([[1,0,0],
                    [1,0.5,0],
                    [1,1,0],
                    [1,1,0.67],
                    [1,1,1],
                    [0.67,0.67,1],
                    [0.25,0.25,1]])  # Corresponding colors
    
    if Ts <= Tval[0]:
        S_color = rgb[0]
    elif Ts >= Tval[-1]:
        S_color = rgb[-1]
    else:
        hi_arg = np.where(Ts < Tval)[0][0]
        lw_arg = hi_arg - 1
        S_color = (rgb[lw_arg] + rgb[hi_arg])/2
    return S_color

def system_visual(wave,Ms,Rs,Ts,a,Rp,b,Ag,AB,varep,Porb,T0,Td,Tn):
    plt.figure(figsize=(12,6))
    ax = plt.gca()
    ax.set_axis_bgcolor('k')
    plt.xticks([])
    plt.yticks([])
    plt.xlim(-3,9)
    plt.ylim(-3,3)
    stx,plx = 0,6
    tsize = 24
    
    s_rad = 2.4  # Keep this fixed (using 2.0 for now) and let planet size vary
    rsjup = Rs*9.955  # Converts to Jupiter radii
    p_rad = (Rp/rsjup)*s_rad  # Scales planet size correctly
    vert_pos = s_rad*b  # Places planet correctly, given impact parameter
    
    ## Colors from temperatures
    d_color = np.asarray(cm.inferno(((Td/T0)**4.0)/(2.0/3.0)))
    if Ag <= 1:
        ag_color = (Ag,Ag,Ag)
    else:
        ag_color = (2-Ag,1,1)  # Since Ag should be 1.5 at most
    n_color = np.asarray(cm.inferno(((Tn/T0)**4.0)/(2.0/3.0)))
    S_color = star_T_color(Ts)
    
    ## Star/planet patches
    star = pat.Circle((stx,0),radius=s_rad,lw=1,ec='0.25',fc=S_color,fill=True,zorder=1)
    ax.add_patch(star)
    sm_ang = np.arctan2(b*s_rad,plx-stx)*180.0/pi  # Small angle to rotate wedges by (aligning dayside correctly)
    day = pat.Wedge((plx,vert_pos),r=p_rad,theta1=90+sm_ang,theta2=270+sm_ang,lw=1,ec='0.25',
                    fc=d_color,fill=True,zorder=1)
    ax.add_patch(day)
    cloud = pat.Wedge((plx,vert_pos),r=1.5*p_rad,theta1=90+sm_ang,theta2=270+sm_ang,width=0.3*p_rad,lw=1,ec=ag_color,
                    fc=ag_color,fill=True,zorder=1)
    ax.add_patch(cloud)
    night = pat.Wedge((plx,vert_pos),r=p_rad,theta1=-90+sm_ang,theta2=90+sm_ang,lw=1,ec='0.25',
                      fc=n_color,fill=True,zorder=1)
    ax.add_patch(night)
    
    ## Important lines
    plt.scatter(stx,0,s=20,marker='o',c='k',zorder=2)
    plt.scatter(stx,vert_pos,s=60,marker='x',c='k',zorder=3)
    cross_lng = s_rad*((1 - b**2.0)**0.5)
    plt.plot([stx+cross_lng,plx],[vert_pos,vert_pos],c='w',lw=2,ls='--',zorder=0)
    plt.plot([stx,stx+cross_lng],[vert_pos,vert_pos],c='k',lw=2,ls='--',zorder=2)
    plt.plot([stx,stx],[0,vert_pos],c='k',lw=3,ls=':',zorder=1)
    
    ## Text
    shift,M_vert = 0.7*s_rad,0.5*(s_rad+3)
    ax.text(stx-shift,M_vert,'$T_{s}=%i \ K$' % Ts,color='w',size=tsize,ha='center',va='center')
    ax.text(stx+shift,M_vert,'$M_{s}=%.2f \ M_{\odot}$' % Ms,color='w',size=tsize,ha='center',va='center')
    ax.text(stx,-M_vert,r'$R_{s}=%.2f \ R_{\odot}$' % Rs,color='w',size=tsize,ha='center',va='center')
    ax.text(stx-0.5*shift,0.5*vert_pos,'$b=%.2f$' % b,color='k',size=tsize,ha='center',va='center')
    a_hor,a_vert = 0.5*((stx+s_rad)+(plx-p_rad)),vert_pos-0.1*s_rad
    ax.text(a_hor,a_vert,'$a=%.3f \ \mathrm{AU}$' % a,color='w',size=tsize,ha='center',va='center')
    ax.text(a_hor,-0.5*M_vert,'$T_{0}=%i \ K$' % T0,color='w',size=tsize,ha='center',va='center')
    ax.text(a_hor,-0.75*M_vert,'$P_{\mathrm{orb}}=%.2f \ \oplus \mathrm{days}$' % Porb,
            color='w',size=tsize,ha='center',va='center')
    ax.text(a_hor,-M_vert,'$R_{p}=%.2f \ R_{\mathrm{Jupiter}}$' % Rp,
            color='w',size=tsize,ha='center',va='center')
    ax.text(plx+shift,-0.5*M_vert,'$A_{g}=%.2f$' % Ag,color='w',size=tsize,ha='center',va='center')
    ax.text(plx+shift,-0.75*M_vert,'$A_{B}=%.2f$' % AB,color='w',size=tsize,ha='center',va='center')
    ax.text(plx+shift,-M_vert,r'$\mathbb{\varepsilon}=%.2f$' % varep,color='w',size=tsize,ha='center',va='center')
    if wave == 'Bolo':
        ax.text(plx+shift,M_vert,r'$\mathrm{Bolometric}$',color='w',size=tsize,ha='center',va='center')
    else:
        ax.text(plx+shift,M_vert,r'$\lambda=%.2f \ \mu\mathrm{m}$' % wave,color='w',size=tsize,ha='center',va='center')
    ax.text(stx-0.85*s_rad,-0.85*s_rad,'$\leftarrow$\n$\mathrm{Observer}$',color='w',size=0.75*tsize,ha='center',va='center')
    
    plt.tight_layout()
    plt.show()
    return

def LC_plotter(blind,hilite,zoom,t_vz,data,a_mod,d_mod,f_mod,brown,bwn_mod,edep,sig_e,D_e):
    if hilite == True:
        alp = 0.1
    else:
        alp = 1.0

    if zoom == True:
        lwid = 1
    else:
        lwid = 2

    plt.figure(figsize=(12,6))
    plt.scatter(t_vz,data,c='k',s=1,alpha=0.75)
    if blind == False:
        plt.plot(t_vz,a_mod,c='b',lw=lwid)
        plt.plot(t_vz,d_mod,c='r',lw=0.5*lwid,alpha=alp)
        plt.plot(t_vz,f_mod,c='m',lw=0.5*lwid,alpha=alp)
        plt.axhline(1.0,c=(0.75,0.75,0),ls='--',lw=2)
        if brown == True:
            plt.plot(t_vz,bwn_mod,c=(0.5,0.25,0),lw=lwid,alpha=alp)
        if zoom == True:
            plt.ylim(1.0 - 2.0*edep,1.0 + 2.0*edep)
    else:
        if zoom == True:
            hi_here,lw_here = np.amax(data),2.0*np.amax([np.mean(data),np.median(data)]) - np.amax(data)
            plt.ylim(lw_here - 0.1*(hi_here - lw_here),hi_here + 0.1*(hi_here - lw_here))
    plt.xlim(0,t_vz[-1])
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    plt.xlabel('Time (hrs)',size='xx-large')
    plt.ylabel('Normalized Flux',size='xx-large')
    plt.title(r'Light Curve - $\mathbb{S}_{e} = %.2f$ - $\Delta D_{e} \approx %.2f$' %
              (sig_e,D_e),size=24);

    plt.tight_layout()
    plt.show()
    return

def full_pixelsens(vx_A,vy_A,Vm,contrs,xo,yo):
    plt.figure(figsize=(7,6))

    sens = plt.contourf(vx_A,vy_A,Vm,contrs,cmap=cm.magma)
    s_bar = plt.colorbar(sens)
    s_bar.formatter.set_useOffset(False)
    s_bar.update_ticks()
    plt.scatter(xo,yo,color='w',s=1)
    plt.xticks(np.linspace(12.5,18.5,7))
    plt.yticks(np.linspace(12.5,18.5,7))
    plt.grid(True,which='major',color='w')
    plt.xlabel('$x_{0}$',size=24);
    plt.ylabel('$y_{0}$',size=24);
    plt.title('Pixel Sensitivity Map',size=24);
    plt.xlim([np.amin(vx_A),np.amax(vx_A)])
    plt.ylim([np.amin(vy_A),np.amax(vy_A)])

    plt.tight_layout()
    plt.show()
    return

def cent_mapplot(xo,yo,vx,vy):  # vx and vy are vectors here- don't use the full arrays vx_A and vy_A like above
    edgex,edgey = np.zeros(len(vx)+1),np.zeros(len(vy)+1)
    dx,dy = vx[1]-vx[0],vy[1]-vy[0]
    
    edgex[:-1],edgey[:-1] = vx - 0.5*dx,vy - 0.5*dy
    edgex[-1],edgey[-1] = vx[-1] + 0.5*dx,vy[-1] + 0.5*dy
    
    counts,ig1,ig2 = np.histogram2d(xo,yo,[edgex,edgey])
    bad_mask = (np.transpose(counts) == 0)

    lxo,hxo,lyo,hyo = np.amin(xo),np.amax(xo),np.amin(yo),np.amax(yo)
    return bad_mask,lxo,hxo,lyo,hyo

def cent_sensfigs(t_vz,xo,yo,lxo,hxo,lyo,hyo,Vm,mp_bmask,bounds,mnv,mxv):
    plt.figure(figsize=(12,6))

    plt.subplot2grid((2,5),(0,0),rowspan=1,colspan=2)
    plt.plot(t_vz,xo,'0.25',lw=0.5)
    plt.xlim([0,t_vz[-1]]);
    plt.gca().set_xticklabels([])
    plt.ylabel('Centroid x',size='x-large');
    plt.ylim([lxo-0.025,0.025+hxo])
    plt.locator_params(axis='y',nbins=5)

    plt.subplot2grid((2,5),(1,0),rowspan=1,colspan=2)
    plt.plot(t_vz,yo,'0.25',lw=0.5)
    plt.xlim([0,t_vz[-1]]);
    plt.xlabel('Time (hr)',size='x-large');
    plt.ylabel('Centroid y',size='x-large');
    plt.ylim([lyo-0.025,0.025+hyo])
    plt.locator_params(axis='y',nbins=5)

    plt.subplot2grid((2,5),(0,2),rowspan=2,colspan=3)
    plt.scatter(xo,yo,color='k',alpha=0.1,s=3,marker='.')

    Vc_map = np.copy(Vm)
    Vc_map[mp_bmask] = np.nan

    my_Smap = plt.imshow(Vc_map,interpolation='hermite',origin='lower',
               extent=bounds,cmap=cm.viridis,vmin=mnv,vmax=mxv)
    Smap_bar = plt.colorbar(my_Smap,label='Sensitivity',extend='both',shrink=1.0)
    Smap_bar.formatter.set_useOffset(False)
    Smap_bar.update_ticks()

    for i in np.linspace(12.5,18.5,7):
        plt.axvline(i,c='0.5',ls='--')
        plt.axhline(i,c='0.5',ls='--')

    plt.gca().set_aspect((hxo-lxo)/(hyo-lyo))
    plt.xlabel('Pixel x',size='x-large');
    plt.ylabel('Pixel y',size='x-large');
    plt.xlim([lxo,hxo])
    plt.ylim([lyo,hyo])
    plt.locator_params(axis='x',nbins=8)
    plt.locator_params(axis='y',nbins=8)

    plt.tight_layout(w_pad=3)
    plt.show()
    return

def examp_Btype_Dm(t_vz,d_mod,bflux):
    plt.figure(figsize=(10,5))
    plt.plot(t_vz,d_mod,c='r',lw=1,alpha=0.5)
    plt.plot(t_vz,bflux,c='b',lw=1,alpha=0.5)
    plt.xlim(0,t_vz[-1])
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    plt.xlabel('Time (hrs)',size='x-large');
    plt.ylabel('Flux',size='x-large');
    plt.title(r'$D(x_{0},y_{0})$ in Red - BLISS in Blue',size=24);

    plt.tight_layout()
    plt.show()
    return

#####

def data_like(theta,t_observe,obs_data,n_data,n_cadence,
              dxyi_lin,n_k,boundkt,
              b_flux,xo,yo,dx,dy,xi,yi,
              lbx,hbx,lby,hby,ll,lr,ul,ur,bls,nni,run_type): # theta has: 5A, 3T, 3E, 1SF, #C (P-type)
    
    astro,trans,ecl,sF = theta[:5],theta[5:8],theta[8:11],theta[11]
    Porb = astro[1]  # period in seconds
    n_per_orb = int(np.around((Porb/3600.0)*n_cadence))  # period in hrs * data per hr
    t_dummy = np.linspace(-0.25*Porb,0.75*Porb,n_per_orb)
    a_model = perf_astro_model(t_observe,t_dummy,astro,trans,ecl)
    
    if run_type == 'P':
        pass
    elif run_type == 'B':
        sens_map = map_flux_avgQuick(obs_data,a_model,dxyi_lin,n_k,boundkt)
        d_model = bliss_meth(n_data,b_flux,sens_map,xo,yo,dx,dy,xi,yi,
                             lbx,hbx,lby,hby,ll,lr,ul,ur,bls,nni)
        
    numer = obs_data - (a_model*d_model)
    lglike = -n_data*np.log(sF) - 0.5*np.sum((numer/sF)**2.0)  # Corrected: include "n_data"
    return lglike

# Probably want to mix uniform and Gaussian priors together
def data_prior(theta,G_key,pri_span):
    if G_key == False:
        if np.all(theta < pri_span[0]) and np.all(theta > pri_span[1]):
            return 0.0
        return -np.inf
    else:
        numer = theta - pri_span[0]
        lgpri = -0.5*np.sum((numer/pri_span[1])**2.0)
        return lgpri

def data_post(theta,G_key,pri_span,
              t_observe,obs_data,n_data,n_cadence,
              dxyi_lin,n_k,boundkt,
              b_flux,xo,yo,dx,dy,xi,yi,
              lbx,hbx,lby,hby,ll,lr,ul,ur,bls,nni,run_type):
    lgpri = data_prior(theta,G_key,pri_span)
    if np.isfinite(lgpri) == False:
        return -np.inf
    return lgpri + data_like(theta,t_observe,obs_data,n_data,n_cadence,
                             dxyi_lin,n_k,boundkt,
                             b_flux,xo,yo,dx,dy,xi,yi,
                             lbx,hbx,lby,hby,ll,lr,ul,ur,bls,nni,run_type)

#####

def labeler(pfit_ord,cn_fit):
    v_lab = np.array([r'$\alpha_{A}$',r'$P_{\mathrm{planet}}$',r'$\phi_{\mathrm{off}}$',r'$\Delta_{A}$',r'$T_{\mathrm{transit}}$',
                      r'$\tau_{t}^{1/2}$',r'$\chi_{t}$',r'$\delta_{t}$',
                      r'$\tau_{e}^{1/2}$',r'$\tau_{\mathrm{ineg}}$',r'$\delta_{e}$',
                      r'$\sigma_{F}$'],dtype='object_')

    pc_lab = np.empty(cn_fit,dtype='object_')
    pcl_i = 0
    for x in np.linspace(pfit_ord,0,pfit_ord+1):
        for y in np.linspace(x,0,x+1):
            if (y == 0) and (pfit_ord-x == 0):
                pass  # Do nothing
            else:
                pc_lab[pcl_i] = '$c_{y=%i}^{x=%i}$' % (y,pfit_ord-x)
                pcl_i += 1
    return v_lab,pc_lab


def mini_thinner(mychain,burn,inc):
    thin_dat = mychain[:,burn::inc,:]  # Better chain thinning
    n_it = thin_dat.shape
    flatdata = thin_dat.reshape((n_it[0]*n_it[1],n_it[2]))
    return flatdata

def AstTransEcl_corner(mychain,burn,inc,v_lab,At,Tt,Et,St):
    flatdata = mini_thinner(mychain,burn,inc)
    fig = corner.corner(flatdata[:,:12],labels=v_lab,label_kwargs={'fontsize':30},
                        truths=np.concatenate((At,Tt,Et,St)))
    plt.show()
    return

def Coeff_corner(mychain,burn,inc,pfit_ord,pc_lab):
    flatdata = mini_thinner(mychain,burn,inc)
    fig = corner.corner(flatdata[:,12:12+pfit_ord],labels=pc_lab[0:pfit_ord],
                        label_kwargs={'fontsize':30})
    plt.show()
    return
    
def AFD_style(mychain,burn,inc,draws,zoom,t_observe,obs_data,f_mod,a_mod,d_mod,bwn_noi,
              n_data,n_cadence,
              dxyi_lin,n_k,boundkt,
              b_flux,xo,yo,dx,dy,xi,yi,
              lbx,hbx,lby,hby,ll,lr,ul,ur,bls,nni,run_type):
    plt.figure(figsize=(12,18))
    t_viz = t_observe/3600
    
    plt.subplot(311)
    plt.scatter(t_viz,obs_data/(d_mod*bwn_noi),c='k',alpha=0.1,zorder=1)
    plt.ylabel('$A(t)$',size=30);
    plt.plot(t_viz,a_mod,'b',linewidth=2,zorder=3)
    plt.xlim([0,t_viz[-1]])
    ahi = np.amax(a_mod)
    if zoom == False:
        alw = np.amin(a_mod)
    elif zoom == True:
        alw = 1.0
    plt.ylim([alw - 0.2*(ahi-alw),ahi + 0.2*(ahi-alw)])
    
    plt.subplot(312)
    plt.scatter(t_viz,obs_data,c='k',alpha=0.1,zorder=1)
    plt.ylabel('$F(t)$',size=30);
    plt.plot(t_viz,f_mod,'m',linewidth=2,zorder=3)
    plt.xlim([0,t_viz[-1]])
    fhi = np.amax(f_mod)
    if zoom == False:
        flw = np.amin(f_mod)
    elif zoom == True:
        flw = np.amin([f_mod[np.argmin(d_mod)],1.0])
    plt.ylim([flw - 0.2*(fhi-flw),fhi + 0.2*(fhi-flw)])
    
    plt.subplot(313)
    plt.scatter(t_viz,obs_data/(a_mod*bwn_noi),c='k',alpha=0.1,zorder=1)
    plt.xlabel('$t$ (hrs)',size=30)
    plt.ylabel('$D(x_{0},y_{0})$',size=30);
    plt.plot(t_viz,d_mod,'r',linewidth=2,zorder=3)
    dlw,dhi = np.amin(d_mod),np.amax(d_mod)
    plt.xlim([0,t_viz[-1]])
    plt.ylim([dlw - 0.2*(dhi-dlw),dhi + 0.2*(dhi-dlw)])
    
    flatdata = mini_thinner(mychain,burn,inc)

    for theta in flatdata[np.random.randint(len(flatdata),size=draws)]:
        astro,trans,ecl,sF = theta[:5],theta[5:8],theta[8:11],theta[11]
        Porb = astro[1]  # period in seconds
        n_per_orb = int(np.around((Porb/3600.0)*n_cadence))  # period in hrs * data per hr
        t_dummy = np.linspace(-0.25*Porb,0.75*Porb,n_per_orb)
        y_ast = perf_astro_model(t_observe,t_dummy,astro,trans,ecl)
        
        plt.subplot(311)
        plt.plot(t_viz,y_ast,'k',alpha=0.1,zorder=2)
        plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
        
        if run_type == 'P':
            plt.subplot(311)
            plt.title('Astro POLY',size=30)
            plt.subplot(312)
            plt.title('Full POLY',size=30)
            plt.subplot(313)
            plt.title('Detector POLY',size=30)
        elif run_type == 'B':
            sens_map = map_flux_avgQuick(obs_data,y_ast,dxyi_lin,n_k,boundkt)
            y_det = bliss_meth(n_data,b_flux,sens_map,xo,yo,dx,dy,xi,yi,
                                 lbx,hbx,lby,hby,ll,lr,ul,ur,bls,nni)
            plt.subplot(311)
            plt.title('Astro BLISS',size=30)
            plt.subplot(312)
            plt.title('Full BLISS',size=30)
            plt.subplot(313)
            plt.title('Detector BLISS',size=30)
            
        plt.subplot(312)
        plt.plot(t_viz,y_ast*y_det,'k',alpha=0.1,zorder=2)
        plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
        plt.subplot(313)
        plt.plot(t_viz,y_det,'k',alpha=0.1,zorder=2)
        plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
        
    plt.tight_layout()
    plt.show()
    return

def walk_style(mychain,ndim,burn,inc,v_lab,pc_lab,At,Tt,Et,St,run_type):  # inc thins out plots
    max_t = mychain.shape[1]
    t_V = np.linspace(1,max_t,max_t)
    t_V = t_V[burn::inc]
    
    if run_type == 'P':
        nrows = np.ceil(ndim/4)
        plt.figure(figsize=(16,3*nrows))
    elif run_type == 'B':
        plt.figure(figsize=(16,9))
        
    for j in np.linspace(0,ndim-1,ndim):
        if run_type == 'P':
            plt.subplot(nrows,4,j+1)
        elif run_type == 'B':
            plt.subplot(3,4,j+1)
        
        mu_param = np.mean(mychain[:,:,j][:,burn::inc],axis=0)
        std_param = np.std(mychain[:,:,j][:,burn::inc],axis=0)
        plt.plot(t_V,mu_param,'k--')
        plt.fill_between(t_V,mu_param + 3.0*std_param,mu_param - 3.0*std_param,facecolor='k',alpha=0.1)
        plt.fill_between(t_V,mu_param + 2.0*std_param,mu_param - 2.0*std_param,facecolor='k',alpha=0.1)
        plt.fill_between(t_V,mu_param + 1.0*std_param,mu_param - 1.0*std_param,facecolor='k',alpha=0.1)
        
        if j <= 11:
            plt.title(v_lab[j],size=16)
        else:
            plt.title(pc_lab[j-12],size=16)
            
        if j < 5:
            plt.plot(np.array([0,max_t]),np.array([At[j],At[j]]),c='b',lw=2)
        elif j < 8:
            plt.plot(np.array([0,max_t]),np.array([Tt[j-5],Tt[j-5]]),c='g',lw=2)
        elif j < 11:
            plt.plot(np.array([0,max_t]),np.array([Et[j-8],Et[j-8]]),c='y',lw=2)
        elif j == 11:
            plt.plot(np.array([0,max_t]),np.array([St[j-11],St[j-11]]),c=(1,0.5,0),lw=2)
        
        if j < (ndim-4):
            plt.xticks([])
        else:
            plt.xticks(rotation=25)

        plt.xlim([burn,max_t])

    plt.tight_layout()
    plt.show()
    return

def dimensions_walkers(r_scl,o_scl,times_dimen,cn_fit,At,Tt,Et,St):
    nB = len(At)+len(Tt)+len(Et)+len(St)
    nP = nB + cn_fit
    if ((times_dimen*nB)/nP) > 4:
        wB = times_dimen*nB
        wP = wB
    else:
        wP = 4*nP
        wB = wP
    
    r_ATES,r_C = np.random.randn(nB),np.random.randn(cn_fit)
    rd_P = np.concatenate((r_ATES,r_C))
    rd_B = r_ATES
    
    o_ATES = np.random.randn(wP*nB).reshape(wP,nB)
    o_C = np.random.randn(wP*cn_fit).reshape(wP,cn_fit)
    
    of_P = np.concatenate((o_ATES,o_C),axis=1)
    of_B = o_ATES
    
    Dt = np.zeros(cn_fit)
    Rl_P = np.concatenate((At,Tt,Et,St,Dt))
    Rl_B = np.concatenate((At,Tt,Et,St))
    
    P_0 = Rl_P*(1.0 + r_scl*rd_P)
    B_0 = Rl_B*(1.0 + r_scl*rd_B)

    P_0Z = P_0*(1.0 + o_scl*of_P)
    B_0Z = B_0*(1.0 + o_scl*of_B)
    
    return nP,nB,wP,wB,P_0Z,B_0Z

#####

def LCdumpGiv(folder,file,Ms,Rs,Ts,
              a,Rp,b,Ag,AB,varep,
              n_hr,obs_orb,frac_trans,wave,bndwid,Se,DDe,
              frac_repnts,nsy_c_key,frac_sigxyo,rho_max,v_noi,nzy_ppix,
              bwn_key,B_DD):
    np.savez(folder+file+'_givens',Ms,Rs,Ts,
             a,Rp,b,Ag,AB,varep,
             n_hr,obs_orb,frac_trans,wave,bndwid,Se,DDe,
             frac_repnts,nsy_c_key,frac_sigxyo,rho_max,v_noi,nzy_ppix,
             bwn_key,B_DD)
    return

def LCdumpCen(folder,file,xo_p,yo_p,t_ind,slct_pnt,
              J_val,W_val,SD_vals,LD_val,
              xo_v,yo_v,cov_tru,xg_fix,yg_F,mast_pnoi):
    np.savez(folder+file+'_cents',xo_p,yo_p,t_ind,slct_pnt,
             J_val,W_val,SD_vals,LD_val,
             xo_v,yo_v,cov_tru,xg_fix,yg_F,mast_pnoi)
    return

def LCdumpDat(folder,file,ast_tru,trans_tru,ecl_tru,mu_tru,sigf_tru,
              tAry,a_mdl,d_mdl,f_perf,f_data,v_map,vx_v,vy_v,bro_noi):
    np.savez(folder+file+'_datas',ast_tru,trans_tru,ecl_tru,mu_tru,sigf_tru,
             tAry,a_mdl,d_mdl,f_perf,f_data,v_map,vx_v,vy_v,bro_noi)
    return

#####

def LCloadGiv(folder,file):
    givens = np.load(folder+file+'_givens.npz')
    nf = len(givens.files)
    
    (Ms,Rs,Ts,
     a,Rp,b,Ag,AB,varep,
     n_hr,obs_orb,frac_trans,wave,bndwid,Se,DDe,
     frac_repnts,nsy_c_key,frac_sigxyo,rho_max,v_noi,nzy_ppix,
     bwn_key,B_DD) = [givens['arr_%i' % j] for j in np.linspace(0,nf-1,nf)]
    
    return (Ms,Rs,Ts,a,Rp,b,Ag,AB,varep,n_hr,obs_orb,frac_trans,wave,bndwid,Se,DDe,
            frac_repnts,nsy_c_key,frac_sigxyo,rho_max,v_noi,nzy_ppix,bwn_key,B_DD)

def LCloadCen(folder,file):
    cents = np.load(folder+file+'_cents.npz')
    nf = len(cents.files)
    
    (xo_p,yo_p,t_ind,slct_pnt,
     J_val,W_val,SD_vals,LD_val,
     xo_v,yo_v,cov_tru,xg_fix,yg_F,mast_pnoi) = [cents['arr_%i' % j] for j in np.linspace(0,nf-1,nf)]
    
    return (xo_p,yo_p,t_ind,slct_pnt,J_val,W_val,SD_vals,LD_val,xo_v,yo_v,cov_tru,xg_fix,yg_F,mast_pnoi)

def LCloadDat(folder,file):
    datas = np.load(folder+file+'_datas.npz')
    nf = len(datas.files)
    
    (ast_tru,trans_tru,ecl_tru,mu_tru,sigf_tru,
     tAry,a_mdl,d_mdl,f_perf,f_data,v_map,vx_v,vy_v,bro_noi) = [datas['arr_%i' % j] for j in np.linspace(0,nf-1,nf)]
    
    return (ast_tru,trans_tru,ecl_tru,mu_tru,sigf_tru,tAry,a_mdl,d_mdl,f_perf,f_data,v_map,vx_v,vy_v,bro_noi)
