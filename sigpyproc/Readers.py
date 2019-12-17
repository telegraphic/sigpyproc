import os
import time
import sigpyproc.HeaderParams as conf
import numpy as np
from inspect import stack as istack
import struct
from struct import unpack
from sys import stdout
from sigpyproc.Utils import File
from sigpyproc.Header import Header
from sigpyproc.Filterbank import Filterbank,FilterbankBlock
from sigpyproc.TimeSeries import TimeSeries
from sigpyproc.FourierSeries import FourierSeries

from sigpyproc.PSRFits import SpectraInfo, unpack_2bit, unpack_4bit

class FilReader(Filterbank):
    """Class to handle the reading of sigproc format filterbank files
    
    :param filename: name of filterbank file
    :type filename: :func:`str`
    
    .. note::
    
       To be considered as a Sigproc format filterbank file the header must only 
       contain keywords found in the ``HeaderParams.header_keys`` dictionary. 
    """
    def __init__(self,filename):
        self.filename = filename
        self.header = parseSigprocHeader(self.filename)
        self._file = File(filename,"r",self.header.nbits)
        self.itemsize = np.dtype(self.header.dtype).itemsize
        if self.header.nbits in [1,2,4]:
            self.bitfact = 8/self.header.nbits
        else:
            self.bitfact = 1
        self.sampsize = int(self.header.nchans*self.itemsize/self.bitfact)
        super(FilReader,self).__init__()

    def readBlock(self,start,nsamps):
        """Read a block of filterbank data.
        
        :param start: first time sample of the block to be read
        :type start: int

        :param nsamps: number of samples in the block (i.e. block will be nsamps*nchans in size)
        :type nsamps: int

        :return: 2-D array of filterbank data
        :rtype: :class:`~sigpyproc.Filterbank.FilterbankBlock`
        """
        self._file.seek(self.header.hdrlen+start*self.sampsize)
        data = self._file.cread(self.header.nchans*nsamps)
        nsamps_read = data.size // self.header.nchans
        data = data.reshape(nsamps_read, self.header.nchans).transpose()
        start_mjd = self.header.mjdAfterNsamps(start)
        new_header = self.header.newHeader({'tstart':start_mjd})
        return FilterbankBlock(data,new_header)
                
    def readPlan(self,gulp,skipback=0,start=0,nsamps=None,verbose=True):
        """A generator used to perform filterbank reading.
 
        :param gulp: number of samples in each read
        :type gulp: int

        :param skipback: number of samples to skip back after each read (def=0)
        :type skipback: int

        :param start: first sample to read from filterbank (def=start of file)
        :type start: int

        :param nsamps: total number samples to read (def=end of file)
        :type nsamps: int

        :param verbose: flag for display of reading plan information (def=True)
        :type verbose: bool

        :return: An generator that can read through the file.
        :rtype: generator object
        
        .. note::

           For each read, the generator yields a tuple ``x``, where:
           
              * ``x[0]`` is the number of samples read
              * ``x[1]`` is the index of the read (i.e. ``x[1]=0`` is the first read)
              * ``x[2]`` is a 1-D numpy array containing the data that was read

           The normal calling syntax for this is function is:

           .. code-block:: python
           
              for nsamps, ii, data in self.readPlan(*args,**kwargs):
                  # do something

           where data always has contains ``nchans*nsamps`` points. 

        """

        if nsamps is None:
            nsamps = self.header.nsamples-start
        if nsamps<gulp:
            gulp = nsamps
        tstart = time.time()
        skipback = abs(skipback)
        if skipback >= gulp:
            raise ValueError("readsamps must be > skipback value")
        self._file.seek(self.header.hdrlen+start*self.sampsize)
        nreads = nsamps//(gulp-skipback)
        lastread = nsamps-(nreads*(gulp-skipback))
        if lastread<skipback:
            nreads -= 1
            lastread = nsamps-(nreads*(gulp-skipback))
        blocks = [(ii,gulp*self.header.nchans,-skipback*self.header.nchans) for ii in range(nreads)]
        blocks.append((nreads,lastread*self.header.nchans,0))
        
        if verbose:
            print()
            print("Filterbank reading plan:")
            print("------------------------")
            print("Called on file:       ",self.filename)      
            print("Called by:            ",istack()[1][3])
            print("Number of samps:      ",nsamps)
            print("Number of reads:      ",nreads)
            print("Nsamps per read:      ",blocks[0][1]/self.header.nchans)
            print("Nsamps of final read: ",blocks[-1][1]/self.header.nchans)
            print("Nsamps to skip back:  ",-1*blocks[0][2]/self.header.nchans)
            print()
        
        for ii,block,skip in blocks:
            if verbose:
                stdout.write("Percentage complete: %d%%\r"%(100*ii/nreads))
                stdout.flush()
            data = self._file.cread(block)
            self._file.seek(skip*self.itemsize//self.bitfact,os.SEEK_CUR)
            yield int(block/self.header.nchans),int(ii),data
        if verbose:
            print("Execution time: %f seconds     \n"%(time.time()-tstart))



class FitsReader(Filterbank):
    """
    Class to handle the reading of PSRFits format fits files
    
    :param filename: list of PSRFits files
    :type filename: :func:`str`
    
    .. note::
    
       To be considered as a PSRFits format fits file the header must be
       readable using Astropy pyfits package. 
    """
    def __init__(self, psrfitslist):
        if isinstance(psrfitslist, str):
            psrfitslist = [psrfitslist]

        psrfitsfn = psrfitslist[0]
        if not os.path.isfile(psrfitsfn):
            raise ValueError("ERROR: File does not exist!\n\t(%s)" % psrfitsfn)
        self.filename = psrfitsfn
        self.filelist = psrfitslist
        self.fileid   = 0
        self._fits    = pyfits.open(psrfitsfn, mode='readonly', memmap=True)

        # Header 
        self.specinfo = SpectraInfo(psrfitslist)
        self.header   = self.specinfo.to_sigpyproc()

        super(FitsReader,self).__init__()


    def readBlock(self, start, nsamps, apply_weights=False, 
                        apply_scales=False, apply_offsets=False):
        """
        Read a block of PSRFits data (and return in filterbank).

        :param start: first time sample of the block to be read
        :type start: int
        :param nsamps: number of samples in the block (i.e. block will be nsamps*nchans in size)
        :type nsamps: int
        :return: 2-D array of filterbank data
        :rtype: :class:`~sigpyproc.Filterbank.FilterbankBlock`
        """
        # Calculate starting subint and ending subint
        startsub = int(start / self.specinfo.spectra_per_subint)
        skip     = int(start - (startsub * self.specinfo.spectra_per_subint))
        endsub   = int((start + nsamps) / self.specinfo.spectra_per_subint)
        trunc    = int(((endsub + 1) * self.specinfo.spectra_per_subint) - (start + nsamps))

        # sort of cread (#TODO if possible)
        # Read full subints (need to be more fast #TODO)
        data     = self.get_data(self, startsub, endsub, apply_weights=apply_weights, 
                                    apply_scales=apply_scales, apply_offsets=apply_offsets)
        # data shape is (nchan, nsample)
        # Truncate data to desired interval
        if trunc > 0:
            data = data[:, skip:-trunc]
        elif trunc == 0:
            data = data[:, skip:]
        else:
            raise ValueError("Number of bins to truncate is negative: %d" % trunc)

        start_mjd  = self.header.mjdAfterNsamps(start)
        new_header = self.header.newHeader({'tstart':start_mjd})
        return FilterbankBlock(data, new_header)



    def readPlan(self,gulp,skipback=0,start=0,nsamps=None,verbose=True):
        """
        A generator used to perform PSRFits reading.
 
        :param gulp: number of samples in each read
        :type gulp: int

        :param skipback: number of samples to skip back after each read (def=0)
        :type skipback: int

        :param start: first sample to read from filterbank (def=start of file)
        :type start: int

        :param nsamps: total number samples to read (def=end of file)
        :type nsamps: int

        :param verbose: flag for display of reading plan information (def=True)
        :type verbose: bool

        :return: An generator that can read through the file.
        :rtype: generator object
        
        .. note::

           For each read, the generator yields a tuple ``x``, where:
           
              * ``x[0]`` is the number of samples read
              * ``x[1]`` is the index of the read (i.e. ``x[1]=0`` is the first read)
              * ``x[2]`` is a 1-D numpy array containing the data that was read

           The normal calling syntax for this is function is:

           .. code-block:: python
           
              for nsamps, ii, data in self.readPlan(*args,**kwargs):
                  # do something

           where data always has contains ``nchans*nsamps`` points. 

        """
        if nsamps is None:
            nsamps = self.header.nsamples-start
        if nsamps<gulp:
            gulp = nsamps

        tstart = time.time()
        skipback = abs(skipback)
        if skipback >= gulp:
            raise ValueError("readsamps must be > skipback value")

        nreads   = nsamps//(gulp-skipback)
        lastread = nsamps-(nreads*(gulp-skipback))
        if lastread<skipback:
            nreads -= 1
            lastread = nsamps-(nreads*(gulp-skipback))
        blocks = [(ii, gulp, -skipback) for ii in range(nreads)]
        blocks.append((nreads, lastread, 0))
        
        if verbose:
            print()
            print("PSRFits reading plan:")
            print("------------------------")
            print("Called on file:       ",self.filename)      
            print("Called by:            ",istack()[1][3])
            print("Number of samps:      ",nsamps)
            print("Number of reads:      ",nreads)
            print("Nsamps per read:      ",blocks[0][1])
            print("Nsamps of final read: ",blocks[-1][1])
            print("Nsamps to skip back:  ",-1*blocks[0][2])
            print()
        
        for ii,block,skipback in blocks:
            if verbose:
                stdout.write("Percentage complete: %d%%\r"%(100*ii/nreads))
                stdout.flush()

            # Calculate starting subint and ending subint
            startsub = int(start / self.specinfo.spectra_per_subint)
            skip     = int(start - (startsub * self.specinfo.spectra_per_subint))
            endsub   = int((start + block) / self.specinfo.spectra_per_subint)
            trunc    = int(((endsub + 1) * self.specinfo.spectra_per_subint) - (start + block))

            # sort of cread (#TODO if possible)
            # Read full subints (need to be more fast #TODO)
            data     = self.get_data(self, startsub, endsub, apply_weights=apply_weights, 
                                    apply_scales=apply_scales, apply_offsets=apply_offsets)

            # data shape is (nchan, nsample)
            # Truncate data to desired interval
            if trunc > 0:
                data = data[:, skip:-trunc]
            elif trunc == 0:
                data = data[:, skip:]
            else:
                raise ValueError("Number of bins to truncate is negative: %d" % trunc)

            start    = start + block + skipback
            data     = data.transpose().ravel()
            yield int(block),int(ii),data

        if verbose:
            print("Execution time: %f seconds     \n"%(time.time()-tstart))




    def get_data(self, startsub, endsub, apply_weights=False, 
                        apply_scales=False, apply_offsets=False):
        """
        #FUTURE Work: Move this whole function in C
        Source: https://github.com/devanshkv/your/blob/master/your/psrfits.py 
        Return 2D array of data from PSRFITS file.

        :param startsub: first subint to be read
        :type startsub: int
        :param endsub: last subint to read
        :type endsub: int
        """      
        cumsum_num_subint = np.cumsum(self.specinfo.num_subint)
        startfileid = np.where(startsub < cumsum_num_subint)[0][0]
        assert startfileid < len(self.filelist)

        if startfileid != self.fileid:
            self.fileid = startfileid
            self._fits.close()
            del self._fits['SUBINT']
            self.filename = self.filelist[self.fileid]
            self._fits = pyfits.open(self.filename, mode='readonly', memmap=True)

        # Read data
        data = []
        for isub in range(startsub, endsub + 1):
            if isub > cumsum_num_subint[self.fileid] - 1:
                self._fits.close()
                del self._fits['SUBINT']
                self.fileid += 1
                if self.fileid == len(self.filelist):
                    self.fileid-=1
                    break
                self.filename = self.filelist[self.fileid]
                self._fits = pyfits.open(self.filename, mode='readonly', memmap=True)

            fsub = int((isub - np.concatenate([np.array([0]),cumsum_num_subint]))[self.fileid])
            try:
                data.append(self.read_subint(fsub, apply_weights=apply_weights, 
                                                   apply_scales=apply_scales, 
                                                   apply_offsets=apply_offsets))
            except KeyError:
                self._fits = pyfits.open(self.filename, mode='readonly', memmap=True)
                data.append(self.read_subint(fsub, apply_weights=apply_weights, 
                                                   apply_scales=apply_scales, 
                                                   apply_offsets=apply_offsets))

        if len(data) > 1:
            data = np.concatenate(data)
        else:
            data = np.array(data).squeeze()
        data = np.transpose(data)                   # (nchan, nsample)
        
        #         if not self.specinfo.need_flipband:
        # TODO

        return data



    def read_subint(self, isub, apply_weights=True, apply_scales=True, \
                    apply_offsets=True):
        """
        Read a PSRFITS subint from a open pyfits file object.
         Applys scales, weights, and offsets to the data.
             Inputs: 
                isub: index of subint (first subint is 0)
                apply_weights: If True, apply weights. 
                    (Default: apply weights)
                apply_scales: If True, apply scales. 
                    (Default: apply scales)
                apply_offsets: If True, apply offsets. 
                    (Default: apply offsets)
             Output: 
                data: Subint data with scales, weights, and offsets
                     applied in float32 dtype with shape (nsamps,nchan).
        """
        sdata = self._fits['SUBINT'].data[isub]['DATA']      #(NSAMP_subint, NPOL, NCHAN)
        shp = sdata.squeeze().shape
        if self.specinfo.bits_per_sample < 8:  # Unpack the bytes data
            if (shp[0] != self.specinfo.spectra_per_subint) and \
                    (shp[1] != self.header.nchans * self.specinfo.bits_per_sample // 8):
                sdata = sdata.reshape(self.specinfo.spectra_per_subint,
                                 int(self.header.nchans * self.specinfo.bits_per_sample // 8))
            if self.specinfo.bits_per_sample == 4:
                data = unpack_4bit(sdata)
            elif self.specinfo.bits_per_sample == 2:
                data = unpack_2bit(sdata)
            else:
                data = np.asarray(sdata)
        else:
            # Handle 4-poln GUPPI/PUPPI data
            if (len(shp) == 3 and shp[1] == self.specinfo.num_polns and
                    self.specinfo.poln_order == "AABBCRCI"):
                logger.warning("Polarization is AABBCRCI, summing AA and BB")
                data = np.zeros((self.specinfo.spectra_per_subint,
                                 self.header.nchans), dtype=np.float32)
                data += sdata[:, 0, :].squeeze()
                data += sdata[:, 1, :].squeeze()
            elif (len(shp) == 3 and shp[1] == self.specinfo.num_polns and
                  self.specinfo.poln_order == "IQUV"):
                logger.warning("Polarization is IQUV, just using Stokes I")
                data = np.zeros((self.specinfo.spectra_per_subint,
                                 self.header.nchans), dtype=np.float32)
                data += sdata[:, 0, :].squeeze()
            else:
                data = np.asarray(sdata)
        data = data.reshape((self.spectra_per_subint,
                             self.header.nchans)).astype(np.float32)
        if apply_scales: data *= self.get_scales(isub)[:self.header.nchans]
        if apply_offsets: data += self.get_offsets(isub)[:self.header.nchans]
        if apply_weights: data *= self.get_weights(isub)[:self.header.nchans]
        return data

    def get_weights(self, isub):
        """
        Return weights for a particular subint.
            Inputs:
                isub: index of subint (first subint is 0)
            
            Output:
                weights: Subint weights. (There is one value for each channel)
        """
        return self._fits['SUBINT'].data[isub]['DAT_WTS']

    def get_scales(self, isub):
        """
        Return scales for a particular subint.
            Inputs:
                isub: index of subint (first subint is 0)
            
            Output:
                scales: Subint scales. (There is one value for each channel)
        """
        return self._fits['SUBINT'].data[isub]['DAT_SCL']

    def get_offsets(self, isub):
        """
        Return offsets for a particular subint.
            Inputs:
                isub: index of subint (first subint is 0)
            
            Output:
                offsets: Subint offsets. (There is one value for each channel)
        """
        return self._fits['SUBINT'].data[isub]['DAT_OFFS']






def readDat(filename,inf=None):
    """Read a presto format .dat file.

    :param filename: the name of the file to read
    :type filename: :func:`str`
    
    :params inf: the name of the corresponding .inf file (def=None)
    :type inf: :func:`str`

    :return: an array containing the whole dat file contents
    :rtype: :class:`~sigpyproc.TimeSeries.TimeSeries`
    
    .. note::

       If inf=None, the function will look for a corresponding file with 
       the same basename which has the .inf file extension.
    """   

    basename = os.path.splitext(filename)[0]
    if inf is None:
        inf = "%s.inf"%(basename)
    if not os.path.isfile(inf):
        raise IOError("No corresponding inf file found")
    header = parseInfHeader(inf)
    f = File(filename,"r",nbits=32)
    data = np.fromfile(f,dtype="float32")
    header["basename"] = basename
    header["inf"] = inf
    header["filename"] = filename
    header["nsamples"] = data.size
    return TimeSeries(data,header)

def readTim(filename):
    """Read a sigproc format time series from file.

    :param filename: the name of the file to read
    :type filename: :func:`str`
    
    :return: an array containing the whole file contents
    :rtype: :class:`~sigpyproc.TimeSeries.TimeSeries`
    """
    header = parseSigprocHeader(filename)
    nbits    = header["nbits"]
    hdrlen   = header["hdrlen"]
    f = File(filename,"r",nbits=nbits)
    f.seek(hdrlen)
    data = np.fromfile(f,dtype=header["dtype"]).astype("float32")
    return TimeSeries(data,header)

def readFFT(filename,inf=None):
    """Read a presto .fft format file.

    :param filename: the name of the file to read
    :type filename: :func:`str`
    
    :params inf: the name of the corresponding .inf file (def=None)
    :type inf: :func:`str`
    
    :return: an array containing the whole file contents
    :rtype: :class:`~sigpyproc.FourierSeries.FourierSeries`

    .. note::

       If inf=None, the function will look for a corresponding file with 
       the same basename which has the .inf file extension.
    """
    basename = os.path.splitext(filename)[0]
    if inf is None:
        inf = "%s.inf"%(basename)
    if not os.path.isfile(inf):
        raise IOError("No corresponding inf file found")
    header = parseInfHeader(inf)
    f = File(filename,"r",nbits=32)
    data = np.fromfile(f,dtype="float32")
    header["basename"] = basename
    header["inf"] = inf
    header["filename"] = filename
    return FourierSeries(data,header)

def readSpec(filename):
    """Read a sigpyproc format spec file.

    :param filename: the name of the file to read
    :type filename: :func:`str`
    
    :return: an array containing the whole file contents
    :rtype: :class:`~sigpyproc.FourierSeries.FourierSeries`

    .. note::

       This is not setup to handle ``.spec`` files such as are
       created by Sigprocs seek module. To do this would require 
       a new header parser for that file format.
    """
    header = parseSigprocHeader(filename)
    hdrlen   = header["hdrlen"]
    f = File(filename,"r",nbits=32)
    f.seek(hdrlen)
    data = np.fromfile(f,dtype="complex32")
    return FourierSeries(data,header)

def parseInfHeader(filename):
    """Parse the metadata from a presto ``.inf`` file.

    :param filename: file containing the header
    :type filename: :func:`str`

    :return: observational metadata
    :rtype: :class:`~sigpyproc.Header.Header`
    """
    f = open(filename,"r")
    header = {}
    lines = f.readlines()
    f.close()
    for line in lines:
        key = line.split("=")[0].strip()
        val = line.split("=")[-1].strip()
        if not key in list(conf.inf_to_header.keys()):
            continue
        else:
            key,keytype = conf.inf_to_header[key]
            header[key] = keytype(val)

    header["src_raj"]      = float("".join(header["src_raj"].split(":")))
    header["src_dej"]      = float("".join(header["src_dej"].split(":")))
    header["telescope_id"] = conf.telescope_ids.get(header["telescope_id"],10)
    header["machine_id"]   = conf.machine_ids.get(header["machine_id"],9)
    header["data_type"]    = 2
    header["nchans"]       = 1
    header["nbits"]        = 32
    header["hdrlen"]       = 0
    header["nsamples"]     = 0
    return Header(header)

def parseSigprocHeader(filename):
    """Parse the metadata from a Sigproc-style file header.

    :param filename: file containing the header
    :type filename: :func:`str`
    
    :return: observational metadata
    :rtype: :class:`~sigpyproc.Header.Header`
    """
    f = open(filename,"rb")
    header = {}
    try:
        keylen = unpack("I",f.read(4))[0]
    except struct.error:
        raise IOError("File Header is not in sigproc format... Is file empty?")
    key = f.read(keylen)
    if key != b"HEADER_START":
        raise IOError("File Header is not in sigproc format")
    while True:
        keylen = unpack("I",f.read(4))[0]
        key = f.read(keylen)
        
        # convert bytestring to unicode (Python 3)
        try:
            key = key.decode("UTF-8")
        except UnicodeDecodeError as e:
            print("Could not convert to unicode: {0}".format(str(e)))

        if not key in conf.header_keys:
            print("'%s' not recognised header key"%(key))
            return None

        if conf.header_keys[key] == "str":
            header[key] = _read_string(f)
        elif conf.header_keys[key] == "I":
            header[key] = _read_int(f)
        elif conf.header_keys[key] == "b":
            header[key] = _read_char(f)
        elif conf.header_keys[key] == "d":
            header[key] = _read_double(f)
        if key == "HEADER_END":
            break

    header["hdrlen"] = f.tell()
    f.seek(0,2)
    header["filelen"]  = f.tell()
    header["nbytes"] =  header["filelen"]-header["hdrlen"]
    header["nsamples"] = int(8*header["nbytes"]/header["nbits"]/header["nchans"])
    f.seek(0)
    header["filename"] = filename
    header["basename"] = os.path.splitext(filename)[0]
    f.close()
    return Header(header) 
        
def _read_char(f):
    return unpack("b",f.read(1))[0]

def _read_string(f):
    strlen = unpack("I",f.read(4))[0]
    return f.read(strlen).decode("UTF-8")

def _read_int(f):
    return unpack("I",f.read(4))[0]

def _read_double(f):
    return unpack("d",f.read(8))[0]

