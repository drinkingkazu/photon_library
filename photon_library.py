import h5py  as h5
import numpy as np
import os
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PhotonLibrary(object):
    '''
    PhotonLibrary class
    The purpose of this class is to provide a probability of observing a photon produced inside a detector.
    The probability is provided per photon detector (N detectors total) for any position inside a detector.
    This information is stored in 2D array of a shape (K,N).
    The detector (rectangular) volume is voxelized in the cartesian coordinate in the total of K voxels.
    The visibility is provided per voxel, and hence it stays same for positions within the same voxel.
    The number of voxels along xyz axis is stored in _npx attribute.
    The voxelized volume boundary in the world coordinate is defined by _min and _max attributes.
    The look up table of shape (K,N) can be found in _vis attribute.

    Two useful attribute functions
    
      - VisibilityXYZ(position,channel=None) returns the visibility value (FP32) given a position and 
        an optical detector identified by channel. If you leave channel unspecified, it returns an array 
        of visibility corresponding to all detectors.
        
      - numpy(ch=None) returns a look-up table in a 3+1D numpy array where dimensions are (x,y,z,ch).
        If ch argument is given to specify a particular optical detector, the returned array will be 3D (x,y,z).
    '''
        
        
    def __init__(self,fname='plib.h5'):
        '''
        Constructor
        INPUT
          fname - the data file that holds the lookup table
        '''

        if not os.path.isfile(fname):
            print('Downloading photon library file... (>300MByte, may take minutes')
            os.system('curl -O https://www.nevis.columbia.edu/~kazuhiro/plib.h5 ./')
        if not os.path.isfile(fname):
            print('Error: failed to download the photon library file...')
            raise Exception

        with h5.File(fname,'r') as f:
            self._vis  = np.array(f['vis'])
            self._min  = np.array(f['min'])
            self._max  = np.array(f['max'])
            self.shape = np.array(f['numvox'])
            
        pmt_data = np.loadtxt('pmt_loc.csv',skiprows=1,delimiter=',')
        if not (pmt_data[:,0].astype(np.int32) == np.arange(pmt_data.shape[0])).all():
            raise Exception('pmt_loc.csv contains optical channel data not in order of channel numbers')
        self._pmt_pos = pmt_data[:,1:4]
        self._pmt_dir = pmt_data[:,4:7]
        if not self._pmt_pos.shape[0] == self._vis.shape[1]:
            raise Exception('Optical detector count mismatch: photon library %d v.s. pmt_loc.csv %d' % (self._vis.shape[1],
                                                                                                        self._pmt_pos.shape[0])
                           )
        if ((self._pmt_pos < self._min) | (self._max < self._pmt_pos)).any():
            raise Exception('Some PMT positions are out of the volume bounds')
        # Convert the PMT positions in a normalized coordinate (fractional position within the voxelized volume)
        self._pmt_pos = (self._pmt_pos - self._min) / (self._max - self._min)
        
        # Run voxelID <=> position self consistency check
        self._ConsistencyCheck_()
        
    
    def __len__(self):
        '''
        Length = the total number of voxels
        '''
        return len(self._vis)
    
    
    def __get__(self,idx):
        '''
        Random access via voxel ID, returns an array of visibility for all PMTs
        '''
        return self._vis[idx]
    
    
    def __str__(self):
        print(self.__class__)
        
        
    def numpy(self,ch=None):
        '''
        Returns a 3+1D numpy array of the visibility look up table, where dimensions correspond to (x,y,z,channel)
        INPUT
          ch - An integer to specify an optical detector ID. If provided, the return will be 3D array
        '''
        shape = [self.shape[2],self.shape[1],self.shape[0]]
        original = self._vis
        if ch is None:
            shape.append(self._vis.shape[1])
        else:
            original = self._vis[:,ch]
            
        return np.swapaxes(original.reshape(shape),0,2)
    
    
    def Visibility(self, vid, ch=None):
        '''
        Returns a probability for a detector to observe a photon.
        If ch (=detector ID) is unspecified, returns an array of probability for all detectors
        INPUT
          vid - Integer voxel ID
          ch  - Integer (valid range 0 to N-1) to specify an optical detector (optional)
        RETURN
          Probability(ies) in FP32 to observe a photon at a specified location.
        '''
        if ch is None:
            return self._vis[vid]
        return self._vis[vid][ch]
    
    
    def VisibilityFromXYZ(self, pos, ch=None):
        '''
        Returns a probability for a detector to observe a photon. See Visibility() for details.
        INPUT
          pos - Length 3 array of FP32 position (x,y,z)
          ch  - Integer (valid range 0 to N-1) to specify an optical detector (optional)
        RETURN
          Probability(ies) in FP32 to observe a photon at a specified location.
        '''
        return self.Visibility(self.Position2VoxID(pos),ch)
        

    def VisibilityFromAxisID(self, axis_id, ch=None):
        '''
        Returns a probability for a detector to observe a photon. See Visibility() for details.
        INPUT
          axis_id - Length 3 array of FP32 position (x,y,z)
          ch  - Integer (valid range 0 to N-1) to specify an optical detector (optional)
        RETURN
          Probability(ies) in FP32 to observe a photon at a specified location.
        '''
        return self.Visibility(self.AxisID2VoxID(axis_id),ch)
    
    
    def UniformSample(self,num_points=32,use_numpy=True,use_world_coordinate=False):
        '''
        Samples visibility for a specified number of points uniformly sampled within the voxelized volume
        INPUT
          num_points - number of points to be sampled
          use_numpy - if True, the return is in numpy array. If False, the return is in torch Tensor
          use_world_coordinate - if True, returns absolute (x,y,z) position. Else fractional position is returned.
        RETURN
          An array of position, shape (num_points,3)
          An array of visibility, shape (num_points,180)
        '''
        
        array_ctor = np.array if use_numpy else torch.Tensor
        
        pos = np.random.uniform(size=num_points*3).reshape(num_points,3)
        axis_id = (pos[:] * self.shape).astype(np.int32)
        
        if use_world_coordinate:
            pos = array_ctor([self.AxisID2Position(apos) for apos in axis_id])
        else:
            pos = array_ctor(pos)
            
        vis = array_ctor([self.VisibilityFromAxisID(apos) for apos in axis_id])

        return pos,vis
        
    
    def _ConsistencyCheck_(self):
        '''
        Simple method to check the consistency
        '''
        if not len(self) == len(self._vis):
            print('Voxel count and xyz index count does not match')
            raise Exception
            
        for vid in (np.random.uniform(size=1000)*len(self._vis)).astype(np.int32):
            pos = self.VoxID2Position(vid)
            if not vid == self.Position2VoxID(pos):
                print('Voxel position <=> ID consistency check failed at position:',pos)
                raise Exception
                

    def AxisID2VoxID(self, axis_id):
        '''
        Takes an integer ID for voxels along xyz axis (ix, iy, iz) and converts to a voxel ID
        INPUT
          axis_id - Length 3 integer array noting the position in discretized index along xyz axis
        RETURN
          The voxel ID (single integer)          
        '''
        return axis_id[0] + axis_id[1]*self.shape[0] + axis_id[2]*(self.shape[0] * self.shape[1])
        
        
    def Position2VoxID(self, pos):
        '''
        Takes a xyz position (x,y,z) and converts to a voxel ID
        INPUT
          pos - Length 3 floating point array noting the position along xyz axis
        RETURN
          The voxel ID (single integer)          
        '''
        axis_id = ((pos - self._min) / (self._max - self._min) * self.shape).astype(np.int32)
        
        if (axis_id < 0).any() or (axis_id >= self.shape).any():
            return -1
        
        return axis_id[0] + axis_id[1]*self.shape[0] + axis_id[2]*(self.shape[0] * self.shape[1])

    
    def VoxID2AxisID(self, vid):
        '''
        Takes a voxel ID and converts to discretized index along xyz axis
        INPUT
          vid - The voxel ID (single integer)          
        RETURN
          Length 3 integer array noting the position in discretized index along xyz axis
        '''
        xid = int(vid) % self.shape[0]
        yid = int((vid - xid) / self.shape[0]) % self.shape[1]
        zid = int((vid - xid - (yid * self.shape[0])) / (self.shape[0] * self.shape[1])) % self.shape[2]
        
        return np.array([xid,yid,zid]).astype(np.float32) 
    
    
    def VoxID2Position(self, vid):
        '''
        Takes a voxel ID and converts to a xyz position (x,y,z)
        INPUT
          vid - The voxel ID (single integer)
        RETURN
          Length 3 floating point array noting the position along xyz axis
        '''
        return self.AxisID2Position(self.VoxID2AxisID(vid))
    
    
    def AxisID2Position(self, axis_id):
        '''
        Takes a axis ID (discretized location along xyz axis) and converts to a xyz position (x,y,z)
        INPUT
          axis_id - The axis ID in an integer array (ix,iy,iz)
        RETURN
          Length 3 floating point array noting the position along xyz axis
        '''    
        return self._min + (self._max - self._min) / self.shape * (axis_id + 0.5)
    
    
    def Position2Features(self, pos, ch=None, use_world_coordinate=False):
        '''
        Takes a voxel ID and returns a geometrical feture vector for a specified optical detector
        INPUT
          vid - The voxel ID (single integer)
          ch  - Optial detector ID (optional, if None, feature vector is made for all detectors in a single array)
        RETURN
          A 2D array of shape (N,2) where two features are (1/r^2, theta). If ch is specified, N=1.
        '''
        if self._pmt_pos is None:
            raise Exception('PMT position information is not available... (check if pmt_pos exists)')
        if not use_world_coordinate:
            assert ((0 <= pos) & (pos <= 1)).all()
        else:
            assert ((self._min <= pos) & (pos <= self._max)).all()
            pos = (pos - self._min) / (self._max - self._min)
        
        diff = pos - self._pmt_pos[:]
        pmt_dir = self._pmt_dir
        if ch is not None:
            diff = diff[ch].reshape([1,-1])
            pmt_dir = self._pmt_dir[ch].reshape([1,-1])
            
        r2 = np.sum(diff**2,axis=1)
        theta = np.arccos(np.sum(diff * pmt_dir,axis=1) / np.sqrt(r2))
        
        return np.column_stack([1./r2,theta])
        

    def BoundaryMin(self):
        '''
        Returns the minimum (x,y,z) value of the voxelized volume
        '''
        return self._min
    
    
    def BoundaryMax(self):
        '''
        Returns the maximum (x,y,z) value of the voxelized volume
        '''
        return self._max
    
    
    def NumVoxels(self):
        '''
        Returns the number of voxels along each axis
        '''
        return self.shape
    
    
    
    def Visibility2D(self, axis, frac, ch=None):
        '''
        Provides a 2D slice of a visibility map at a fractional location along the specified axis
        INPUT
          axis - One of three cartesian axis 'x', 'y', or 'z'
          frac - A floating point value in the range [0,1] to specify the location, in fraction, along the axis
          ch   - An integer or a list of integers specifying a (set of) optical detectors (if not provided, visibility for all detectors are summed)
        RETURN
          2D (XY) slice of a visibility map
        '''
        axis_labels = ['x','y','z']
        ia, ib, itarget = 0,0,0
        if   axis == 'x': itarget, ia, ib = 0,1,2
        elif axis == 'y': itarget, ia, ib = 1,2,0
        elif axis == 'z': itarget, ia, ib = 2,0,1
        else:
            print('axis must be x, y, or z')
            raise ValueError
        
        if frac < 0 or 1.0 < frac:
            print('frac must be between 0.0 and 1.0')
            raise ValueError
            
        loc_target = int(float(frac) * self.shape[itarget] + 0.5)
        result = np.zeros(shape=[self.shape[ia],self.shape[ib]],dtype=np.float32)
        axis_id = [0,0,0]
        chs = np.arange(len(self._vis[0]))
        if ch is not None:
            chs = [ch] if type(ch) == type(int()) else ch
        for loc_a in range(self.shape[ia]):
            for loc_b in range(self.shape[ib]):
                axis_id[itarget] = loc_target
                axis_id[ia]      = loc_a
                axis_id[ib]      = loc_b
                vid = self.AxisID2VoxID(axis_id)
                for ch in chs:
                    result[loc_a][loc_b] += self._vis[vid][ch]
        return result
    
    
    def PlotVisibility2D(self,axis,frac,ch=None):
        '''
        Visualize a 2D slice of a visibility map at a fractional location along the specified axis
        INPUT
          axis - One of three cartesian axis 'x', 'y', or 'z'
          frac - A floating point value in the range [0,1] to specify the location, in fraction, along the axis
          ch   - An integer or a list of integers specifying a (set of) optical detectors (if not provided, visibility for all detectors are summed)
        RETURN
          figure objecta
        '''
        axis_labels = ['x','y','z']
        ia, ib = 0,0
        if   axis == 'x': ia, ib = 1,2
        elif axis == 'y': ia, ib = 2,0
        elif axis == 'z': ia, ib = 0,1
        else:
            print('axis must be x, y, or z')
            raise ValueError
            
        ar=self.Visibility2D(axis,frac,ch)
        pos_range=np.column_stack([self.BoundaryMin(),self.BoundaryMax()])
        extent = np.concatenate([pos_range[ib], pos_range[ia]])
        
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        
        fig,ax=plt.subplots(figsize=(16,12),facecolor='w')
        ax.matshow(ar,norm=mpl.colors.LogNorm(),extent=extent)
        ax.tick_params(axis='both',which='both',labelsize=16,bottom=True,top=False,left=True,right=False,labelleft=True,labelbottom=True)
        ax.set_xlabel('%s [cm]' % axis_labels[ib].upper(),fontsize=20)
        ax.set_ylabel('%s [cm]' % axis_labels[ia].upper(),fontsize=20)
        return fig 