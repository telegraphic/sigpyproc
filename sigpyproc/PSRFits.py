#!/usr/bin/env python

"""
Collect PSRFITS information, emulating behaviour of PRESTO.
Read PSRFITS data.

Patrick Lazarus, May 11, 2010
Updated: Jul 4, 2016 (Scott Ransom)
(https://github.com/scottransom/presto/blob/master/python/presto/psrfits.py)

Updated: Dec 15, 2019 (Devansh A.)
(https://github.com/devanshkv/your/blob/master/your/psrfits.py)

Updated: (Integration to sigpyproc)
Pravir Kumar
"""

import re
import os
import os.path
import logging

import numpy as np
import astropy.time as aptime
import astropy.io.fits as pyfits
from astropy import coordinates, units

import sigpyproc.HeaderParams as conf 
from sigpyproc.Header import Header as SigpyprocHeader


logger = logging.getLogger(__name__)


SECPERDAY = float('86400.0')

# Regular expression for parsing DATE-OBS card's format.
date_obs_re = re.compile(r"^(?P<year>[0-9]{4})-(?P<month>[0-9]{2})-" \
                         "(?P<day>[0-9]{2})T(?P<hour>[0-9]{2}):" \
                         "(?P<min>[0-9]{2}):(?P<sec>[0-9]{2}" \
                         "(?:\.[0-9]+)?)$")



class SpectraInfo:
    def __init__(self, filenames):
        self.filenames = filenames
        self.num_files = len(filenames)
        self.N = 0

        # Initialise a few arrays
        self.start_MJD    = np.empty(self.num_files)
        self.num_subint   = np.empty(self.num_files, dtype=np.int64)
        self.start_subint = np.empty(self.num_files)
        self.start_spec   = np.empty(self.num_files, dtype=np.int64)
        self.num_pad      = np.empty(self.num_files, dtype=np.int64)
        self.num_spec     = np.empty(self.num_files, dtype=np.int64)

        # The following should default to False
        self.need_scale    = False
        self.need_offset   = False
        self.need_weight   = False
        self.need_flipband = False

        for ii, fn in enumerate(filenames):
            if not is_PSRFITS(fn):
                raise ValueError("File '%s' does not appear to be PSRFITS!" % fn)

            # Open the PSRFITS file
            with pyfits.open(fn, mode='readonly', memmap=True) as hdus:
                if ii == 0:
                    self.hdu_names = [hdu.name for hdu in hdus]
                primary = hdus['PRIMARY'].header
                
                if 'TELESCOP' not in primary.keys():
                    telescope = ""
                else:
                    telescope = primary['TELESCOP']
                if ii == 0:
                    self.telescope = telescope
                else:
                    if telescope != self.telescope[0]:
                        logger.warning(
                            f"'TELESCOP' values don't match for files 0 ({self.telescope[0]}) and {ii}({telescope})!")

                self.observer   = primary['OBSERVER']
                self.source     = primary['SRC_NAME']
                self.frontend   = primary['FRONTEND']
                self.backend    = primary['BACKEND']
                self.project_id = primary['PROJID']
                self.date_obs   = primary['DATE-OBS']    # UTC
                self.poln_type  = primary['FD_POLN']
                self.ra_str     = primary['RA']
                self.dec_str    = primary['DEC']
                self.fctr       = primary['OBSFREQ']
                self.orig_num_chan = primary['OBSNCHAN']
                self.orig_df    = primary['OBSBW']
                self.beam_FWHM  = primary['BMIN']

                # CHAN_DM card is not in earlier versions of PSRFITS
                if 'CHAN_DM' not in primary.keys():
                    self.chan_dm = 0.0
                else:
                    self.chan_dm = primary['CHAN_DM']

                self.start_MJD[ii] = primary['STT_IMJD'] + (primary['STT_SMJD'] + \
                                                            primary['STT_OFFS']) / SECPERDAY

                # Are we tracking
                track = (primary['TRK_MODE'] == "TRACK")
                if ii == 0:
                    self.tracking = track
                else:
                    if track != self.tracking:
                        logger.warning("'TRK_MODE' values don't match for files 0 and %d" % ii)

                # Now switch to the subint HDU header
                subint = hdus['SUBINT'].header

                self.dt           = subint['TBIN']
                self.num_channels = subint['NCHAN']
                self.num_polns    = subint['NPOL']
                self.poln_order   = subint['POL_TYPE']

                # Channel/sub-band offset for split files
                try:
                	if subint['NCHNOFFS'] > 0:
                    	logger.warning("first freq channel is not 0 in file %d" % ii)
                except TypeError as err:
                	logger.warning("NCHNOFFS is not properly defined in header.")

                self.spectra_per_subint = subint['NSBLK']
                self.bits_per_sample    = subint['NBITS']
                self.num_subint[ii]     = subint['NAXIS2']
                self.start_subint[ii]   = subint['NSUBOFFS']   # Subint offset
                self.time_per_subint    = self.dt * self.spectra_per_subint

                # ZERO_OFF card is not in earlier versions of PSRFITS
                if 'ZERO_OFF' not in subint.keys():
                    self.zero_offset = 0.0
                else:
                    self.zero_offset = subint['ZERO_OFF']

                # This is the MJD offset based on the starting subint number
                MJDf = (self.time_per_subint * self.start_subint[ii]) / SECPERDAY
                # The start_MJD values should always be correct
                self.start_MJD[ii] += MJDf

                # Compute the starting spectra from the times
                MJDf = self.start_MJD[ii] - self.start_MJD[0]
                if MJDf < 0.0:
                    raise ValueError("File %d seems to be from before file 0!" % ii)

                self.start_spec[ii] = (MJDf * SECPERDAY / self.dt + 0.5)

                # Now pull stuff from the columns
                subint_hdu   = hdus['SUBINT']
                first_subint = subint_hdu.data[0]

                # Get the time offset column info and the offset for the 1st row
                # Identify the OFFS_SUB column number
                if 'OFFS_SUB' not in subint_hdu.columns.names:
                    logger.warning("Can't find the 'OFFS_SUB' column!")
                else:
                    colnum = subint_hdu.columns.names.index('OFFS_SUB')
                    if ii == 0:
                        self.offs_sub_col = colnum
                    elif self.offs_sub_col != colnum:
                        logger.warning("'OFFS_SUB' column changes between files 0 and %d!" % ii)

                # Identify the data column and the data type
                if 'DATA' not in subint_hdu.columns.names:
                    logger.warning("Can't find the 'DATA' column!")
                else:
                    colnum = subint_hdu.columns.names.index('DATA')
                    if ii == 0:
                        self.data_col = colnum
                        self.FITS_typecode = subint_hdu.columns[self.data_col].format[-1]
                    elif self.data_col != colnum:
                        logger.warning("'DATA' column changes between files 0 and %d!" % ii)

                # Telescope azimuth
                if 'TEL_AZ' not in subint_hdu.columns.names:
                    self.azimuth = 0.0
                else:
                    colnum = subint_hdu.columns.names.index('TEL_AZ')
                    if ii == 0:
                        self.tel_az_col = colnum
                        self.azimuth = first_subint['TEL_AZ']

                # Telescope zenith angle
                if 'TEL_ZEN' not in subint_hdu.columns.names:
                    self.zenith_ang = 0.0
                else:
                    colnum = subint_hdu.columns.names.index('TEL_ZEN')
                    if ii == 0:
                        self.tel_zen_col = colnum
                        self.zenith_ang = first_subint['TEL_ZEN']

                # Observing frequencies
                if 'DAT_FREQ' not in subint_hdu.columns.names:
                    logger.warning("Can't find the channel freq column, 'DAT_FREQ'!")
                else:
                    colnum = subint_hdu.columns.names.index('DAT_FREQ')
                    freqs = first_subint['DAT_FREQ'][:self.num_channels]
                    if ii == 0:
                        self.freqs_col = colnum
                        self.df        = freqs[1] - freqs[0]
                        # Assuming freq can not be zero.               
                        self.lo_freq   = freqs.min()
                        self.hi_freq   = freqs.max()
                        # Now check that the channel spacing is the same throughout
                        ftmp = freqs[1:] - freqs[:-1]
                        if np.any((ftmp - self.df)) > 1e-7:
                            logger.warning("Channel spacing changes in file %d!" % ii)
                    else:
                        ftmp = np.abs(self.df - (freqs[1] - freqs[0]))
                        if ftmp > 1e-7:
                            logger.warning("Channel spacing between files 0 and %d!" % ii)
                        ftmp = np.abs(self.lo_freq - freqs.min())
                        if ftmp > 1e-7:
                            logger.warning("Low channel changes between files 0 and %d!" % ii)
                        ftmp = np.abs(self.hi_freq - freqs.max())
                        if ftmp > 1e-7:
                            logger.warning("High channel changes between files 0 and %d!" % ii)

                # Data weights
                if 'DAT_WTS' not in subint_hdu.columns.names:
                    logger.warning("Can't find the channel weights column, 'DAT_WTS'!")
                else:
                    colnum = subint_hdu.columns.names.index('DAT_WTS')
                    if ii == 0:
                        self.dat_wts_col = colnum
                    elif self.dat_wts_col != colnum:
                        logger.warning("'DAT_WTS column changes between files 0 and %d!" % ii)
                    if np.any(first_subint['DAT_WTS'][:self.num_channels] != 1.0):
                        self.need_weight = True

                # Data offsets
                if 'DAT_OFFS' not in subint_hdu.columns.names:
                    logger.warning("Can't find the channel offsets column, 'DAT_OFFS'!")
                else:
                    colnum = subint_hdu.columns.names.index('DAT_OFFS')
                    if ii == 0:
                        self.dat_offs_col = colnum
                    elif self.dat_offs_col != colnum:
                        logger.warning("'DAT_OFFS column changes between files 0 and %d!" % ii)
                    if np.any(first_subint['DAT_OFFS'] != 0.0):
                        self.need_offset = True

                # Data scalings
                if 'DAT_SCL' not in subint_hdu.columns.names:
                    logger.warning("Can't find the channel scalings column, 'DAT_SCL'!")
                else:
                    colnum = subint_hdu.columns.names.index('DAT_SCL')
                    if ii == 0:
                        self.dat_scl_col = colnum
                    elif self.dat_scl_col != colnum:
                        logger.warning("'DAT_SCL' column changes between files 0 and %d!" % ii)
                    if np.any(first_subint['DAT_SCL'] != 1.0):
                        self.need_scale = True

                # Comute the samples per file and the amount of padding
                # that the _previous_ file has
                self.num_pad[ii]  = 0
                self.num_spec[ii] = self.spectra_per_subint * self.num_subint[ii]
                if ii > 0:
                    if self.start_spec[ii] > self.N:  # Need padding
                        self.num_pad[ii - 1] = self.start_spec[ii] - self.N
                        self.N += self.num_pad[ii - 1]
                self.N += self.num_spec[ii]

        # Finished looping through PSRFITS files. Finalise a few things.
        # Convert the position strings into degrees        
        self.ra2000 = coordinates.Angle(self.ra_str, unit=units.hourangle).deg
        self.dec2000 = coordinates.Angle(self.dec_str, unit=units.deg).deg

        # Are the polarisations summed?
        if (self.poln_order == "AA+BB") or (self.poln_order == "INTEN"):
            self.summed_polns = True
        else:
            self.summed_polns = False

        # Calculate some others
        self.T        = self.N * self.dt
        self.orig_df /= float(self.orig_num_chan)
        self.samples_per_spectra = self.num_polns * self.num_channels
        # Note: the following is the number of bytes that will be in
        #       the returned array.
        if self.bits_per_sample < 8:
            self.bytes_per_spectra = self.samples_per_spectra
        else:
            self.bytes_per_spectra = (self.bits_per_sample * self.samples_per_spectra) / 8
        self.samples_per_subint = self.samples_per_spectra * self.spectra_per_subint
        self.bytes_per_subint = self.bytes_per_spectra * self.spectra_per_subint

        # Flip the band?
        if self.df < 0:
            #self.df *= -1.0
            self.need_flipband = True

        # Compute the bandwidth
        self.BW   = self.num_channels * self.df
        self.mjd  = int(self.start_MJD[0])
        self.secs = (self.start_MJD[0] % 1) * SECPERDAY

    def __str__(self):
        """
        Format spectra_info's information into a easy to
        read string and return it.
        """
        result = []  # list of strings. Will be concatenated with newlines (\n).
        result.append("From the PSRFITS file '%s':" % self.filenames[0])
        result.append("                       HDUs = %s" % ', '.join(self.hdu_names))
        result.append("                  Telescope = %s" % self.telescope)
        result.append("                   Observer = %s" % self.observer)
        result.append("                Source Name = %s" % self.source)
        result.append("                   Frontend = %s" % self.frontend)
        result.append("                    Backend = %s" % self.backend)
        result.append("                 Project ID = %s" % self.project_id)
        result.append("            Obs Date String = %s" % self.date_obs)
        imjd, fmjd = DATEOBS_to_MJD(self.date_obs)
        mjdtmp = "%.14f" % fmjd
        result.append("  MJD start time (DATE-OBS) = %5d.%14s" % (imjd, mjdtmp[2:]))
        result.append("     MJD start time (STT_*) = %19.14f" % self.start_MJD[0])
        result.append("                   RA J2000 = %s" % self.ra_str)
        result.append("             RA J2000 (deg) = %-17.15g" % self.ra2000)
        result.append("                  Dec J2000 = %s" % self.dec_str)
        result.append("            Dec J2000 (deg) = %-17.15g" % self.dec2000)
        result.append("                  Tracking? = %s" % self.tracking)
        result.append("              Azimuth (deg) = %-.7g" % self.azimuth)
        result.append("           Zenith Ang (deg) = %-.7g" % self.zenith_ang)
        result.append("          Polarisation type = %s" % self.poln_type)
        if (self.num_polns >= 2) and (not self.summed_polns):
            numpolns = "%d" % self.num_polns
        elif self.summed_polns:
            numpolns = "2 (summed)"
        else:
            numpolns = "1"
        result.append("            Number of polns = %s" % numpolns)
        result.append("          Polarisation oder = %s" % self.poln_order)
        result.append("           Sample time (us) = %-17.15g" % (self.dt * 1e6))
        result.append("         Central freq (MHz) = %-17.15g" % self.fctr)
        result.append("          Low channel (MHz) = %-17.15g" % self.lo_freq)
        result.append("         High channel (MHz) = %-17.15g" % self.hi_freq)
        result.append("        Channel width (MHz) = %-17.15g" % self.df)
        result.append("         Number of channels = %d" % self.num_channels)
        if self.chan_dm != 0.0:
            result.append("   Orig Channel width (MHz) = %-17.15g" % self.orig_df)
            result.append("    Orig Number of channels = %d" % self.orig_num_chan)
            result.append("    DM used for chan dedisp = %-17.15g" % self.chan_dm)
        result.append("      Total Bandwidth (MHz) = %-17.15g" % self.BW)
        result.append("         Spectra per subint = %d" % self.spectra_per_subint)
        result.append("            Starting subint = %d" % self.start_subint[0])
        result.append("           Subints per file = %d" % self.num_subint[0])
        result.append("           Spectra per file = %d" % self.num_spec[0])
        result.append("        Time per file (sec) = %-.12g" % (self.num_spec[0] * self.dt))
        result.append("              FITS typecode = %s" % self.FITS_typecode)
        result.append("                DATA column = %d" % self.data_col)
        result.append("            bits per sample = %d" % self.bits_per_sample)
        if self.bits_per_sample < 8:
            spectmp = (self.bytes_per_spectra * self.bits_per_sample) / 8
            subtmp = (self.bytes_per_subint * self.bits_per_sample) / 8
        else:
            spectmp = self.bytes_per_spectra
            subtmp = self.bytes_per_subint
        result.append("          bytes per spectra = %d" % spectmp)
        result.append("        samples per spectra = %d" % self.samples_per_spectra)
        result.append("           bytes per subint = %d" % subtmp)
        result.append("         samples per subint = %d" % self.samples_per_subint)
        result.append("                zero offset = %-17.15g" % self.zero_offset)
        result.append("                                 ")        
        result.append("              Need scaling? = %s" % self.need_scale)
        result.append("              Need offsets? = %s" % self.need_offset)
        result.append("              Need weights? = %s" % self.need_weight)
        result.append("        Need band inverted? = %s" % self.need_flipband)

        return '\n'.join(result)

    def __getitem__(self, key):
        return getattr(self, key)


    def to_sigpyproc(self):
        """
        Header conversion
        """
        header = {}

        # dictionary with filterbank header keys and values.
        header["telescope_id"]  = conf.telescope_ids.get(self.telescope, 0)
        header["machine_id"]    = conf.machine_ids.get(self.backend,0)
        header["data_type"]     = 1                    # filterbank
        header["source_name"]   = self.source
        header["barycentric"]   = 0                    # always not barycentered?
        header["pulsarcentric"] = 0                    # whats pulsarcentric?        
        header["az_start"]      = self.azimuth         # Telescope azimuth
        header["za_start"]      = self.zenith_ang      # Telescope zenith angle
        header["src_raj"]       = float(self.ra_str.replace(':',''))
        header["src_dej"]       = float(self.dec_str.replace(':',''))
        header["tstart"]        = self.start_MJD[0]
        header["tsamp"]         = self.dt
        header["nifs"]          = 1                    # Always 1 polarization (self.num_polns)
        # Always read psrfits data in 32 bits
        header["nbits"]         = 32                   # self.bits_per_sample
        header["nsamples"]      = int(self.N)
        # first channel (fch1) in sigproc is the highest freq
        # foff is negative to signify this
        header["fch1"]          = self.fctr + np.abs(self.num_channels*self.df)/2.0 \
                                            - np.abs(self.df)/2.0
        header["foff"]          = -1.0*np.abs(self.df)
        header["nchans"]        = self.num_channels
        header["rawdatafile"]   = os.path.basename(self.filenames[0])

        return SigpyprocHeader(header)



def DATEOBS_to_MJD(dateobs):
    """
    Convert DATE-OBS string from PSRFITS primary HDU to a MJD.
        Returns a 2-tuple:
            (integer part of MJD, fractional part of MJD)
    """
    # Parse string using regular expression defined at top of file
    m = date_obs_re.match(dateobs)
    mjd_fracday = (float(m.group("hour")) + (float(m.group("min")) + \
                                             (float(m.group("sec")) / 60.0)) / 60.0) / 24.0
    mjd_day = aptime.Time("%d-%d-%d" % (float(m.group("year")), \
                                        float(m.group("month")), float(m.group("day"))), format="iso").mjd
    return mjd_day, mjd_fracday


def is_PSRFITS(filename):
    """
    Return True if filename appears to be PSRFITS format.
        Return False otherwise.
    """
    with pyfits.open(filename, mode='readonly', memmap=True) as hdus:
        primary = hdus['PRIMARY'].header

        try:
            isPSRFITS = ((primary['FITSTYPE'] == "PSRFITS") and \
                         (primary['OBS_MODE'] == "SEARCH"))
        except KeyError:
            isPSRFITS = False

    return isPSRFITS



def unpack_2bit(data):
    """
    Unpack 2-bit data that has been read in as bytes.
        Input: 
            data2bit: array of unsigned 2-bit ints packed into
                an array of bytes.
        Output: 
            outdata: unpacked array. The size of this array will 
                be four times the size of the input data.
    """
    piece0 = np.bitwise_and(data >> 0x06, 0x03)
    piece1 = np.bitwise_and(data >> 0x04, 0x03)
    piece2 = np.bitwise_and(data >> 0x02, 0x03)
    piece3 = np.bitwise_and(data, 0x03)
    return np.dstack([piece0, piece1, piece2, piece3]).flatten()


def unpack_4bit(data):
    """
    Unpack 4-bit data that has been read in as bytes.
        Input: 
            data4bit: array of unsigned 4-bit ints packed into
                an array of bytes.
        Output: 
            outdata: unpacked array. The size of this array will 
                be twice the size of the input data.
    """
    piece0 = np.bitwise_and(data >> 0x04, 0x0F)
    piece1 = np.bitwise_and(data, 0x0F)
    return np.dstack([piece0, piece1]).flatten()
