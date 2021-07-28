"""
Synchronize beacon pings across hydrophones using linear 
interpolation and a direct matrix-based solution.
Soundspeed is limited to piecewise constant.

Usage: 

    all_detections=pd.read_csv('all_detections.csv')
    # hydros.csv preferably has position previously estimated by YAPS.
    hydros=pd.read_csv(os.path.join(data_dir,'hydros.csv'),index_col='serial')

    sp=SyncPings(all_detections,hydros)
    sp.solve()

    t_data=all_detections.epo.max() - all_detections.epo.min()
    print("%d detections. %d bad pings"%
          (len(all_detections), len(sp.bad_pings)))

    # Figure showing timeseries of sound speed, clock offset, counts of
    # synchronization pings, and distibution of residual.
    fig=sp.figure_sync()
    fig.savefig(os.path.join(data_dir,'sync_summary.png'),dpi=250)

    sp.write_synced(os.path.join(data_dir,'all_detections_sync.csv'))
"""

import matplotlib.pyplot as plt
import numpy as np

def dist(a,b):
    """
    distance between points
    """
    vec=np.asarray(a)-np.asarray(b)
    return np.sqrt( (vec**2).sum(axis=-1) )

def combine_detects(tag_detects,slop_s=10.0):
    """
    combine received pings with the same tag into one ping when
    separate by less than slop_s
    """
    det_hydros=tag_detects['hydro'].values-1 # to 0-based
    det_t_fracs=tag_detects['t_frac'].values
    toa_rows=[]
    breaks=np.nonzero( np.diff(det_t_fracs)> slop_s)[0]

    breaks=np.r_[ 0,1+breaks,len(det_t_fracs)]
    for b_start,b_stop in zip(breaks[:-1],breaks[1:]):
        slc=slice(b_start,b_stop)

        toa_row=np.nan*np.zeros(len(hydros))
        toa_row[det_hydros[slc]]=det_t_fracs[slc]

        if np.isfinite(toa_row).sum() < b_stop-b_start:
            # now we can do more expensive testing
            mean_t_frac=np.nanmean(det_t_fracs[slc])
            for h in range(len(hydros)):
                if (det_hydros[slc]==h).sum()>1:
                    # t_frac for the colliding detections
                    t_frac_hydro=det_t_fracs[slc][ det_hydros[slc]==h ]
                    # which one is closest to the mean
                    best=np.argmin(np.abs(t_frac_hydro - mean_t_frac))
                    toa_row[h]=t_frac_hydro[best]
                    print("Discarding a ping")

        toa_rows.append(toa_row)
    return np.array(toa_rows)

class SyncPings(object):
    n_offset_per_day = 8 # How many points per day to fit for the clock offset
    n_ss_per_day = 8     # How many points per day to fit for the sound speed

    ss_default=1450.0 # Default value for soundspeed.
    
    def __init__(self,detections,hydros,tk_serial='AM8.8'):
        """
        hydros: columns serial,x,y,z,idx
        detections: epo, frac
        tk_serial: serial number of the hydrophone to use as a reference
        """
        self.hydros=hydros
        self.detections=detections
        self.T0=int(self.detections['epo'].mean())
        self.tk_serial=tk_serial
        
        self.prepare_data()
    def prepare_data(self):
        self.detections['t_frac']=(self.detections['epo']-self.T0) + self.detections['frac']
        
        data={}
        self.nh=len(self.hydros)
        self.tk=self.hydros.loc[self.tk_serial,'idx'] - 1

        self.detections['hydro']=self.hydros.loc[self.detections['serial'],'idx'].values

        # Combine long-form (one ping*receiver per row) into matix form (one ping per row)
        all_toa_rows=[]
        all_sync_tag_idx=[]

        sync_tags=self.hydros['sync_tag'].values
        for sync_tag in sync_tags:
            tag_detects=self.detections[ self.detections['tag']==sync_tag ].sort_values('t_frac')
            tag_detects=tag_detects.reset_index()
            print(f"tag {sync_tag} with {len(tag_detects)} detections total")
            toa_rows_tag=combine_detects(tag_detects)
            all_toa_rows.append(toa_rows_tag)
            sync_tag_idx=self.hydros[ self.hydros['sync_tag']==sync_tag ]['idx'].values[0]
            all_sync_tag_idx.append( sync_tag_idx * np.ones(toa_rows_tag.shape[0]))

        self.toa_absolute=np.concatenate( all_toa_rows, axis=0)
        # This is 1-based, to match the idx column of hydros
        self.sync_tag_idx=np.concatenate( all_sync_tag_idx ).astype(np.int32)
        assert self.sync_tag_idx.min()>=1
        assert self.sync_tag_idx.max()<=self.nh

        # parameters which will vary during the iterative processing go into
        # data
        self.H= self.hydros.loc[:, ['x','y','z']].values

        # +-5 is overkill, but make sure that all pings fall neatly within the
        # range of offset_breaks.
        t_min=self.detections['t_frac'].min() - 5 # np.nanmin(self.toa_absolute)-30
        t_max=self.detections['t_frac'].max() + 5 # np.nanmax(self.toa_absolute)+30

        # Last thing is the offset and ss periods.
        # Lay it all out globally, and will solve it incrementally
        n_days=(t_max-t_min)/86400.
        # offsets at the global level
        self.n_offset_idx=int( n_days*self.n_offset_per_day+0.4999 )
        self.n_ss_idx=int( n_days*self.n_ss_per_day+0.4999 )

        self.offset_breaks=np.linspace(t_min,t_max,self.n_offset_idx+1)
        self.ss_breaks    =np.linspace(t_min,t_max,self.n_ss_idx+1)

        self.toa_absolute_mean=np.nanmean(self.toa_absolute,axis=1)

        self.offset_idx=np.searchsorted(self.offset_breaks,self.toa_absolute_mean) - 1
        self.toa_offset=self.toa_absolute - self.offset_breaks[self.offset_idx,None]
        self.ss_idx=np.searchsorted(self.ss_breaks,self.toa_absolute_mean) - 1
        # self.sigma_toa=0.0001

        self.dist_mat=self.calc_distances()

        # Which hydrophones actually get offset information
        self.off_mask=np.arange(self.nh)!=self.tk
        # a per-ping toa offset
        self.mean_toa_offset=np.nanmean(self.toa_offset,axis=1)

    def calc_distances(self):
        dist_mat=np.zeros( (self.nh,self.nh), np.float64)
        for h1 in range(self.nh):
            for h2 in range(self.nh):
                dist_mat[h1,h2]=dist(self.H[h1],self.H[h2])
        return dist_mat
        
    def solve(self):
        self.offsets=offsets=np.zeros((self.nh,self.n_offset_idx+1),
                                      np.float64)
        self.sss=sss=np.zeros(self.n_ss_idx,np.float64)
        self.pings_per_interval=np.zeros((self.nh,self.n_offset_idx),
                                         np.int32)

        idxs=np.arange(len(offsets))
        
        assert self.n_offset_idx==self.n_ss_idx,"Not ready for difffering intervals"

        # winsize=4 # 4 intervals => 5 offset vals, 4 ss vals
        # overlap=1
        # 0 ---- 1 ---- 2 ---- 3 -xx- 4
        #                      0 -xx- 1 ---- 2
        # Say there's no data in interval xx
        # then 4 is meaningless in the 1st window, and
        # 0 is meaningless in the 2nd window.

        winsize=6
        overlap=2
        # 0 ---- 1 ---- 2 ---- 3 ---- 4 ---- 5 ---- 6
        #                             0 ---- 1 ---- 2 ----

        start=0
        
        bad_pings=[]
        
        while start < self.n_offset_idx:
            stop=min(start+winsize,self.n_offset_idx)
            
            offset_slc=ss_slc=slice(int(start),
                                    int(stop))
            sub_result=self.solve_subset(offset_slc,ss_slc)
            
            if start==0:
                trim_start=0
            else:
                trim_start=(overlap+1)//2

            # drop the first bit since the previous window has a better
            # fit there.
            offsets[:,start+trim_start:stop+1]=sub_result['offset'][:,trim_start:]
            sss[start+trim_start:stop]=sub_result['ss'][trim_start:]

            self.pings_per_interval[:,start+trim_start:stop] = sub_result['interval_n_pings'][:,trim_start:]
            bad_pings.append(sub_result['bad_pings'])
                
            start+=winsize-overlap

        self.bad_pings=np.unique(np.concatenate(bad_pings))
            
    def solve_subset(self,offset_slc,ss_slc):
        """
        offset_slc: slice of offsets to consider. currently must
        have stride of 1.
        likewise for ss_slc.

        if offset and ss slicing differ it probably won't do well.

        returns dict with offset, ss and bad pings
        """
        # which pings are we interested in?
        ping_sel=( (self.offset_idx>=offset_slc.start)
                   & (self.offset_idx<offset_slc.stop) )
        nping=ping_sel.sum()
        
        # Attempt a direct matrix solve
        Mrows=[]
        bvals=[]

        nh=self.nh
        n_off=offset_slc.stop-offset_slc.start # self.n_offset_idx
        n_ss=ss_slc.stop - ss_slc.start

        off0=offset_slc.start
        ss0=ss_slc.start
        
        # piecewise linear -- n_off intervals => n_off+1 values
        ncols=(n_off+1)*nh + n_ss

        row_pings=[]

        ping_sel_idxs=np.nonzero(ping_sel)[0]
        # p is global ping index
        for p in ping_sel_idxs:
            off_idx = self.offset_idx[p]
            ss_idx = self.ss_idx[p]
            tag_idx=self.sync_tag_idx[p]-1

            off_alpha=( self.mean_toa_offset[p]
                        /
                        (self.offset_breaks[off_idx+1] - self.offset_breaks[off_idx]) )

            hi = np.nonzero( np.isfinite(self.toa_offset[p,:] ))[0]
            h1=hi[0]
            for h2 in hi[1:]:
                row=np.zeros( ncols, np.float64)
                row[ (n_off+1)*h1 + off_idx  -off0] = -(1.0-off_alpha)
                row[ (n_off+1)*h1 + off_idx+1-off0] = -(    off_alpha)
                row[ (n_off+1)*h2 + off_idx  -off0] =  (1-off_alpha)
                row[ (n_off+1)*h2 + off_idx+1-off0] =       off_alpha

                # soundspeed rows have inverse, so they are second / meter
                # sparse3 change:
                row[ nh*(n_off+1) + ss_idx - ss0] = (self.dist_mat[h1,tag_idx] - self.dist_mat[h2,tag_idx])

                Mrows.append(row)
                bvals.append( self.toa_offset[p,h1] - self.toa_offset[p,h2] )
                row_pings.append(p)

        # so a row represents the difference in time-of-flight for a ping
        # from tag_idx to get to h1 vs. h2
        M=np.array(Mrows)
        b=np.array(bvals)
        row_pings=np.array(row_pings)

        # That includes values for the timekeeper
        # drop those columns.
        sel=np.ones(M.shape[1],np.bool8)
        sel[self.tk*(n_off+1):(self.tk+1)*(n_off+1)]=False
        Mslim=M[:,sel]

        bad_rows=[]
        rmse=100

        while len(bad_rows)<200:
            valid=np.ones(Mslim.shape[0],np.bool8)
            valid[bad_rows]=False

            Mslim_nobad=Mslim[valid,:]

            # dense matrix approach
            self.mat_inputs=(Mslim[valid,:],b[valid]) # for some testing
            soln,res,rank,sing=np.linalg.lstsq(Mslim[valid,:],
                                               b[valid],
                                               rcond=-1)

            Merrors=np.zeros(Mslim.shape[0],np.float64)
            Merrors[valid]=Mslim[valid,:].dot(soln) - b[valid]

            rmse=np.sqrt( (Merrors**2).mean() )
            print("RMSE: ",rmse)
            if rmse<0.001:
                print("Good enough")
                break

            # Mark the worst row as bad.
            bad_rows.append( np.argmax( np.abs(Merrors) ) )
            print("Worst ping row had error=",Merrors[bad_rows[-1]])

            print(bad_rows)
        else:
            raise Exception("Failed to get rid of enough bad pings to get a good RMSE")

        # reexpand offsets
        offset=soln[:(nh-1)*(n_off+1)].reshape( (nh-1,n_off+1) )
        offset_exp=np.zeros( (nh,n_off+1), np.float64 )
        not_tk=np.arange(nh)!=self.tk
        offset_exp[not_tk,:]=offset

        bad_pings=np.unique(row_pings[bad_rows])
        ping_sel[bad_pings]=False

        # should have shape [nh,n_off]
        n_pings=[]
        for h in range(self.nh):
            per_h=np.bincount(self.offset_idx[ping_sel],
                              weights=np.isfinite(self.toa_offset[ping_sel,h]),
                              minlength=n_off)
            n_pings.append(per_h[off0:])

        return dict(offset=offset_exp,
                    ss=soln[(nh-1)*(n_off+1):],
                    interval_n_pings=np.array(n_pings),
                    bad_pings=bad_pings)

    def evaluate_sync(self):
        """
        Evaluate a sync model that has already been calculated.

        Calculate the sync parameters for each ping, and some summary
        stats.
        """
        ping_mask=np.ones( len(self.mean_toa_offset), np.bool8)
        ping_mask[self.bad_pings]=False
        
        calcs={}
        off_idx=self.offset_idx
        off_alpha=( self.mean_toa_offset
                    /
                    (self.offset_breaks[off_idx+1] - self.offset_breaks[off_idx]) )
        offset_vals=( (1-off_alpha)*self.offsets[:,off_idx]
                      +
                      off_alpha*self.offsets[:,off_idx+1] )[:,ping_mask]

        calcs['offset_vals']=offset_vals
        calcs['ss_vals']=(1./self.sss[self.ss_idx])[ping_mask]

        # [np,nh] giving transit time
        transit_times = self.dist_mat[self.sync_tag_idx[ping_mask]-1,:] / calcs['ss_vals'][:,None]

        toa_adjusted = self.toa_offset[ping_mask,:] + offset_vals.T
        top_estimate = toa_adjusted - transit_times

        errors=np.nanvar(top_estimate,axis=1)
        calcs['errors']=errors
        calcs['rms_error']=np.sqrt(errors.mean())
        calcs['ping_mask']=ping_mask
        return calcs
        
    # Visualize the result:
    def figure_sync(self,num=1):
        plt.figure(num).clf()
        fig,axs=plt.subplots(4,1,num=num)

        calcs=self.evaluate_sync()

        order=np.argsort(self.toa_absolute_mean[calcs['ping_mask']])

        toa=self.toa_absolute_mean[calcs['ping_mask']][order]
        
        axs[0].plot(toa,
                    calcs['ss_vals'][order], 'g',ms=1)

        mid_offsets=0.5*(self.offset_breaks[:-1] + self.offset_breaks[1:])
        for h in range(self.nh):
            axs[1].plot(toa, calcs['offset_vals'][h,order],label="H=%d"%h)
            axs[2].plot(mid_offsets, self.pings_per_interval[h],label="H=%d"%h)
            
        axs[3].hist(np.sqrt(calcs['errors']),
                    bins=np.linspace(0.0,0.005,100))
        axs[3].set_xlabel('Time-of-flight error (s)')

        axs[3].text(0.2,0.95,"rms error: %.4f"%calcs['rms_error'],
                    transform=axs[3].transAxes,va='top')
        return fig

    def write_synced(self,detections_fn):
        # Add an eposync field
        det=self.detections.copy()
        det.rename(columns={'hydro':'hydro_idx'},inplace=True)

        h=det['hydro_idx']-1

        off_idx=np.searchsorted(self.offset_breaks,det['t_frac'].values) -1
        assert off_idx.max() <= len(self.offset_breaks)-1,"Were t_min/t_max not generous enough?"
        
        toa_offset=self.detections['t_frac'] - self.offset_breaks[off_idx]
        
        ss_idx=np.searchsorted(self.ss_breaks,self.detections['t_frac'].values) - 1
        
        off_alpha=( toa_offset
                    /
                    (self.offset_breaks[off_idx+1] - self.offset_breaks[off_idx]) )
        offset_vals=( (1-off_alpha)*self.offsets[h,off_idx]
                      +
                      off_alpha*self.offsets[h,off_idx+1] )

        ss_vals=(1./self.sss[ss_idx])

        det['epofrac']=det['t_frac'] + self.T0 
        det['eposync']=det['epofrac'] + offset_vals 
        det['ss']=ss_vals
        det.to_csv(detections_fn,index=False)

