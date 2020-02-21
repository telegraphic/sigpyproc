import os, sys
import timeit
import numpy as np
from tqdm import tqdm
from natsort import natsorted
from sigpyproc.Readers import FitsReader, FilReader


def get_fileformat(infiles):
    """
    Find the fileformat based on extension of input files.
    This is not a comprehensive check of data format.
    """
    extension = [os.path.splitext(infile)[-1] for infile in infiles]
    if len(list(set(extension))) > 1:
        raise TypeError(f"Input files do not have same extension.!")
    extension = list(set(extension))[0]
    if extension in (".fits", ".sf"):
        fileformat = "psrfits"
    elif extension == ".fil":
        fileformat = "filterbank"
    else:
        raise TypeError(f"Filetype '{extension}' not supported")
    return fileformat


def files_sanity_check(fits_fns):

    basename = [os.path.splitext(os.path.split(fits_fn)[-1])[0].rsplit('_', 1)[0] \
                        for fits_fn in fits_fns]
    index    = [os.path.splitext(os.path.split(fits_fn)[-1])[0].rsplit('_', 1)[-1] \
                        for fits_fn in fits_fns]
    # files should belong to same pointing/observation
    if len(list(set(basename))) > 1:
        raise TypeError(f'Input files do not belongs to same pointing.')

    # check whether any file is missing
    # Assuming that starting and ending files are fixed 
    # (need to be sorted before this)
    try:
        missing_id  = list(set(index) ^ set([x \
                           for x in range(index[0], index[-1] + 1)]))
        if len(missing_id) > 1:
            raise ValueError(f'Input files are missing for '
                             f'obs-index: {",".join(str(item) for item in missing_id)}.')
    except: pass



def scale_data(myReader, nbits, mode="digifil"):
    """
    Get scale factor when converting summed AA+BB data back to 8 bits.
    """
    if mode == "presto":
        print("\nCalculating statistics on first subintegration...")
        subint0 = myReader.read_subint(0, apply_weights=False, apply_scales=False, 
                                          apply_offsets=False)
        new_max = 3 * np.median(subint0)
        print("\t3*median =", new_max)
        if new_max > 2.0**nbits:
            scale_fac = new_max / ( 2.0**nbits )
            print(f'\tScaling data by {1/scale_fac:.3f}')
            print(f'\tValues larger than {new_max:.3f}(pre-scaling) will be set to {2**nbits - 1}\n')
        else:
            scale_fac = 1
            print(f'\tNo scaling necessary')
            print(f'\tValues larger than {2**nbits-1}(2^nbits) will be set to {2**nbits-1}\n')

    if mode == "digifil":
        print("\nUsing digifil scalings...")
        scale_fac = np.sqrt(2.0)
        print(f'\tScaling data by {1/scale_fac:.3f}')
        print(f'\tValues larger than {(2**nbits-1)*scale_fac:.3f}(pre-scaling) will be set to {2**nbits - 1}\n')

    return scale_fac

def fits_converter(fits_fns, outfn, nbits, nsub, apply_weights=False, 
                   apply_scales=False, apply_offsets=False, mode="digifil"):
    """
    Convert psrfits data to SIGPROC filterbank files. 
    """
    start_time = timeit.default_timer()
    # Check if the files belong to same observations
    if len(fits_fns) > 1:
        fits_fns = natsorted(fits_fns)
        files_sanity_check(fits_fns)

    myFits = FitsReader(fits_fns)
    out_files   = []
    hdr_changes = {}
    chanpersub  = myFits.header.nchans // nsub
    for isub in range(nsub):
        hdr_changes["nchans"] = chanpersub
        hdr_changes["fch1"]   = myFits.header.fch1 + isub*chanpersub*myFits.header.foff
        if nsub > 1:
            outfil = f'{outfn.split(".fil")[0]}_sub{str(isub).zfill(2)}.fil'
        else:
            outfil = f'{outfn.split(".fil")[0]}.fil'
        out_files.append(myFits.header.prepOutfile(outfil, hdr_changes, nbits=nbits))

    scale_fac = scale_data(myFits, nbits=nbits, mode=mode)

    print("PSRFits to Filterbank conversion:")
    print("---------------------------------")
    for nsamps, ii, data in tqdm(
                            myFits.readPlan(gulp=myFits.specinfo.spectra_per_subint, 
                                            verbose=False, 
                                            apply_weights=apply_weights, 
                                            apply_scales=apply_scales, 
                                            apply_offsets=apply_offsets),
                            total=int(myFits.specinfo.num_subint.sum())):
        # scaling, clipping and bit conversion
        if nbits == 8: 
            data /= scale_fac
            data  = data.round().clip(min=0, max=255)  # np.iinfo(data.dtype).max
            data  = data.astype("ubyte", copy=False)

        for isub, out_file in enumerate(out_files):
            data = data.reshape(nsamps, myFits.header.nchans)
            subint_tofil = data[:,chanpersub*isub:chanpersub*(isub+1)]
            out_file.cwrite(subint_tofil.ravel())

    # Now close each of the filterbank file.
    for out_file in out_files:
        out_file.close()

    end_time = timeit.default_timer()
    print(f'Done')
    print(f'Execution time: {(end_time-start_time):.3f} seconds\n')

def fil_splitter(fil_fns, outfn, nsub):
    """
    Split filterbank data to different subband files
    """
    start_time = timeit.default_timer()
    if len(fil_fns) > 1:
        raise TypeError(f'Input files: "{len(fil_fns)}". Not supported yet!.')
    if nsub == 1:
        raise TypeError(f'Nsub: {nsub}. No need for conversion!.')

    myFil = FilReader(fil_fns[0])
    out_files   = []
    hdr_changes = {}
    chanpersub  = myFil.header.nchans // nsub
    for isub in range(nsub):
        hdr_changes["nchans"] = chanpersub
        hdr_changes["fch1"]   = myFil.header.fch1 + isub*chanpersub*myFil.header.foff
        outfil = f'{outfn.split(".fil")[0]}_sub{str(isub).zfill(2)}.fil'
        out_files.append(myFil.header.prepOutfile(outfil, hdr_changes, nbits=myFil.header.nbits))

    print("Splitting Filterbank:")
    print("---------------------------------")
    gulpsize = 32768
    for nsamps, ii, data in tqdm(
                            myFil.readPlan(gulp=gulpsize, verbose=False),
                            total=myFil.header.nsamples//gulpsize):
        for isub, out_file in enumerate(out_files):
            data = data.reshape(nsamps, myFil.header.nchans)
            subint_tofil = data[:,chanpersub*isub:chanpersub*(isub+1)]
            out_file.cwrite(subint_tofil.ravel())

    # Now close each of the filterbank file.
    for out_file in out_files:
        out_file.close()

    end_time = timeit.default_timer()
    print(f'Done')
    print(f'Execution time: {(end_time-start_time):.3f} seconds\n')


#https://astropy.readthedocs.io/en/latest/development/scripts.html
def main():
    import argparse
    description = "Convert to SIGPROC filterbank data files."
    parser = argparse.ArgumentParser(\
                      description = description, \
                      formatter_class = lambda prog: argparse.HelpFormatter(\
                                      prog, max_help_position=100, width = 250))
    parser.add_argument("-f", dest = "infiles", metavar = "", type = str, 
                        help = "input file names (PSRfits or Filterbank)", nargs='+')
    parser.add_argument("-o", dest = "outfn", metavar = "", type = str, action = "store", 
                        help = "Filename (base) of the output filterbank file.")
    parser.add_argument("-n", dest = "nbits", metavar = "", type = int,
                        action = "store", default = 8,
                        help = "Number of bits in the output .fil file. (default: 8)")
    parser.add_argument("-nsub", dest = "nsub", metavar = "", type = int, 
                        action = "store", default = 1, 
                        help = "Number of subbands to split file into (default: 1)")

    parser.add_argument("-scale_mode", dest = "scale_mode", metavar = "", type = str, 
                        default = "digifil", 
                        help = "Scaling type for fits conversion [digifil (default), presto]")
    parser.add_argument("--apply_weights", dest='apply_weights', 
                        action = "store_true", 
                        help="Apply weights when converting data.")
    parser.add_argument("--apply_scales", dest='apply_scales',  
                        action = "store_true", 
                        help = "Apply scales when converting data")
    parser.add_argument("--apply_offsets", dest='apply_offsets', 
                        action = "store_true",
                        help = "Apply offsets when converting data")
    args = parser.parse_args()

    if (not args.infiles) or (not args.outfn):
        print('Input and output paths are required.')
        print(parser.print_help())
        sys.exit(0)

    infile_format = get_fileformat(args.infiles)

    if infile_format == "psrfits":
        fits_converter(args.infiles, args.outfn, args.nbits, args.nsub, args.apply_weights, 
                       args.apply_scales, args.apply_offsets, args.scale_mode)

    elif infile_format == "filterbank":
        fil_splitter(args.infiles, args.outfn, args.nsub)


if __name__=="__main__":
    main()


