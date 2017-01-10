from setuptools import setup
from distutils.core import setup
from distutils.extension import Extension
import numpy
import os

# Set to false to enable tracking of Cython lines in profile
release = True

# Check for ReadTheDocs flag
RTDFLAG = bool(os.environ.get('READTHEDOCS', None) == 'True')

# Check for Cython
try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

ext_options = dict(language="c++",
                   include_dirs=[numpy.get_include()],
                   libraries=[],
                   extra_link_args=[],
                   extra_compile_args=["-std=gnu++11"])

# CYTHON_TRACE required for coverage and line_profiler.  Remove for release.
if not release and use_cython:
    ext_options['define_macros'] = [('CYTHON_TRACE', '1')]

if RTDFLAG:
    ext_options['libraries'] = []
    ext_options['extra_link_args'] = []
    ext_options['extra_compile_args'].append('-DREADTHEDOCS')
    compile_parallel = False
else:
    # Check for existence of mpi
    compile_parallel = True
    try:
        ext_options_mpi = copy.deepcopy(ext_options)
        # Attempt to call MPICH first, then OpenMPI
        try:
            mpi_compile_args = os.popen(
                "mpic++ -compile_info").read().strip().split(' ')[1:]
            mpi_link_args = os.popen(
                "mpic++ -link_info").read().strip().split(' ')[1:]
            if len(mpi_compile_args[0]) == 0:
                raise Exception
        except:
            mpi_compile_args = os.popen(
                "mpic++ --showme:compile").read().strip().split(' ')
            mpi_link_args = os.popen(
                "mpic++ --showme:link").read().strip().split(' ')
            if len(mpi_compile_args[0]) == 0:
                raise Exception
        ext_options_mpi['extra_compile_args'] += mpi_compile_args
        ext_options_mpi['extra_link_args'] += mpi_link_args
    except:
        compile_parallel = False

# Needed for line_profiler - disable for production code
if not RTDFLAG and not release and use_cython:
    try:
        from Cython.Compiler.Options import directive_defaults
    except ImportError:
        # Update to cython
        from Cython.Compiler.Options import get_directive_defaults
        directive_defaults = get_directive_defaults()
    directive_defaults['profile'] = True
    directive_defaults['linetrace'] = True
    directive_defaults['binding'] = True

cmdclass = { }
ext_modules = [ ]
src_include = [ ]

def make_cpp(cpp_file):
    if not os.path.isfile(cpp_file):
        open(cpp_file,'a').close()
        assert(os.path.isfile(cpp_file))

make_cpp("cykdtree/c_kdtree.cpp")
make_cpp("cykdtree/c_utils.cpp")
if compile_parallel:
    make_cpp("cykdtree/c_parallel_kdtree.cpp")

ext_modules += [
    Extension("cykdtree/kdtree",
              sources=["cykdtree/kdtree.pyx",
                       "cykdtree/c_kdtree.cpp",
                       "cykdtree/c_utils.cpp"],
              **ext_options),
    Extension("cykdtree/utils",
              sources=["cykdtree/utils.pyx",
                       "cykdtree/c_utils.cpp"],
              **ext_options)]
if compile_parallel:
    ext_modules.append(
        Extension("cykdtree/parallel_kdtree",
                  sources=["cykdtree/parallel_kdtree.pyx",
                           "cykdtree/c_parallel_kdtree.cpp",
                           "cykdtree/c_kdtree.cpp"],
                  **ext_options_mpi))

src_include += [
    "cykdtree/kdtree.pyx", "cykdtree/c_kdtree.hpp",
    "cykdtree/utils.pyx", "cykdtree/c_utils.hpp",
    "cykdtree/parallel_kdtree.pyx", "cykdtree/c_parallel_kdtree.hpp"]

if use_cython:
    ext_modules = cythonize(ext_modules)
    cmdclass.update({ 'build_ext': build_ext })

with open('README.rst') as file:
    long_description = file.read()

setup(name='cykdtree',
      packages=['cykdtree'],
      package_dir={'cykdtree':'cykdtree'},
      package_data = {'cykdtree': ['README.md', 'README.rst'] + src_include},
      version='0.1.7',
      description='Cython based KD-Tree',
      long_description=long_description,
      author='Meagan Lang',
      author_email='langmm.astro@gmail.com',
      url='https://langmm@bitbucket.org/langmm/cykdtree',
      keywords=['domain decomposition', 'decomposition', 'kdtree'],
      classifiers=["Programming Language :: Python",
                   "Programming Language :: C++",
                   "Operating System :: OS Independent",
                   "Intended Audience :: Science/Research",
                   "License :: OSI Approved :: BSD License",
                   "Natural Language :: English",
                   "Topic :: Scientific/Engineering",
                   "Topic :: Scientific/Engineering :: Astronomy",
                   "Topic :: Scientific/Engineering :: Mathematics",
                   "Topic :: Scientific/Engineering :: Physics",
                   "Development Status :: 3 - Alpha"],
      license='BSD',
      zip_safe=False,
      cmdclass = cmdclass,
      ext_modules = ext_modules,
      data_files = [('cykdtree', src_include)])


