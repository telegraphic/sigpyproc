from setuptools import setup, Extension #find_packages

__version__ = '0.1.1'

def describe(filename):
    f = open(filename, "r")
    lines = f.readlines()
    return "".join(lines)


ext0 = Extension('MersenneTwister',
                 sources=['./c_src/MersenneTwister.c'],
                 extra_link_args=["-lm"],
                 extra_compile_args=["-Wno-strict-prototypes"]
                 )

ext1 = Extension('libSigPyProc8',
                 sources=[
                     './c_src/libSigPyProc8.c',
                     './c_src/MersenneTwister.c',
                 ],
                 extra_link_args=["-lgomp", "-lm"],
                 extra_compile_args=["-fopenmp", "-Wno-unused-variable", "-Wno-strict-prototypes"],
                 )

ext2 = Extension('libSigPyProc32',
                 sources=[
                     './c_src/libSigPyProc32.c',
                 ],
                 extra_link_args=["-lgomp", "-lm"],
                 extra_compile_args=["-fopenmp", "-Wno-unused-variable", "-Wno-strict-prototypes"],
                 )

ext3 = Extension('libSigPyProcSpec',
                 sources=[
                     './c_src/libSigPyProcSpec.c',
                 ],
                 extra_link_args=["-lgomp", "-lm", "-lfftw3", "-lfftw3f"],
                 extra_compile_args=["-fopenmp", "-Wno-unused-variable", "-Wno-strict-prototypes"],
                 )

ext4 = Extension('libSigPyProcTim',
                 sources=[
                     './c_src/libSigPyProcTim.c',
                 ],
                 extra_link_args=["-lgomp", "-lm", "-lfftw3", "-lfftw3f"],
                 extra_compile_args=["-fopenmp", "-Wno-unused-variable", "-Wno-strict-prototypes"],
                 )

ext5 = Extension('libSigPyProc',
                 sources=[
                     './c_src/libSigPyProc.c',
                 ],
                 extra_link_args=["-lgomp"],
                 extra_compile_args=["-fopenmp", "-Wno-unused-variable", "-Wno-strict-prototypes"],
                 )


# http://astropy.readthedocs.org/en/latest/development/scripts.html
entry_points = {
    'console_scripts' : [
        'fil_writer = sigpyproc.fil_writer:main'
     ]
}


install_requires = [
        'astropy',
        'numpy',
        'natsort',
        'tqdm',
]

setup(name='sigpyproc',
      version=__version__,
      description='Python pulsar data toolbox',
      install_requires = install_requires,
      python_requires = '>=3.6',
      author='Ewan Barr',
      author_email='ewan.d.barr@googlemail.com',
      entry_points=entry_points,
      long_description=describe('README.md'),
      ext_modules=[ext0, ext1, ext2, ext3, ext4, ext5],
      packages=['sigpyproc'],
      zip_safe=False
      )
