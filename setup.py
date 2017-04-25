from setuptools import setup
from distutils.extension import Extension
from setuptools.command.sdist import sdist as _sdist
from subprocess import Popen, PIPE
import copy
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
    raise ImportError('Cython is a required dependency of cykdtree')

ext_options = dict(language="c++",
                   include_dirs=[numpy.get_include()],
                   libraries=[],
                   extra_link_args=[],
                   extra_compile_args=["-std=c++03"])


def call_subprocess(args):
    p = Popen(args, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate()
    exit_code = p.returncode
    if exit_code != 0:
        return None
    ret = output.decode().strip().split()
    if ret[0] == 'c++':
        return ret[1:]
    return ret


def get_mpi_args(mpi_executable, compile_argument, link_argument):
    compile_args = call_subprocess([mpi_executable, compile_argument])
    link_args = call_subprocess([mpi_executable, link_argument])
    if compile_args is None:
        return None
    return compile_args, link_args

# CYTHON_TRACE required for coverage and line_profiler.  Remove for release.
if not release:
    ext_options['define_macros'] = [('CYTHON_TRACE', '1')]

ext_options_mpi = copy.deepcopy(ext_options)
if RTDFLAG:
    ext_options['libraries'] = []
    ext_options['extra_link_args'] = []
    ext_options['extra_compile_args'].append('-DREADTHEDOCS')
    compile_parallel = False
else:
    # Check for existence of mpi
    compile_parallel = True
    ret = (
        get_mpi_args('mpic++', '-compile_info', '-link_info') or  # MPICH
        get_mpi_args('mpic++', '--showme:compile', '--showme:link')  # OpenMPI
    )
    if ret is not None:
        mpi_compile_args, mpi_link_args = ret
        ext_options_mpi['extra_compile_args'] += mpi_compile_args
        ext_options_mpi['extra_link_args'] += mpi_link_args

# Needed for line_profiler - disable for production code
if not RTDFLAG and not release:
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
    Extension("cykdtree.kdtree",
              sources=["cykdtree/kdtree.pyx",
                       "cykdtree/c_kdtree.cpp",
                       "cykdtree/c_utils.cpp"],
              **ext_options),
    Extension("cykdtree.utils",
              sources=["cykdtree/utils.pyx",
                       "cykdtree/c_utils.cpp"],
              **ext_options)]
if compile_parallel:
    ext_modules.append(
        Extension("cykdtree.parallel_kdtree",
                  sources=["cykdtree/parallel_kdtree.pyx",
                           "cykdtree/c_parallel_kdtree.cpp",
                           "cykdtree/c_kdtree.cpp",
                           "cykdtree/c_utils.cpp"],
                  **ext_options_mpi))
    print("compiling parallel")

src_include += [
    "cykdtree/kdtree.pyx", "cykdtree/kdtree.pxd", "cykdtree/c_kdtree.hpp",
    "cykdtree/utils.pyx", "cykdtree/utils.pxd", "cykdtree/c_utils.hpp",
    "cykdtree/parallel_kdtree.pyx", "cykdtree/parallel_kdtree.pxd",
    "cykdtree/c_parallel_kdtree.hpp", "cykdtree/c_kdtree.cpp",
    "cykdtree/c_utils.cpp", "cykdtree/c_parallel_kdtree.cpp"]

ext_modules = cythonize(ext_modules)
cmdclass.update({'build_ext': build_ext})

class sdist(_sdist):
    # subclass setuptools source distribution builder to ensure cython
    # generated C files are included in source distribution and readme
    # is converted from markdown to restructured text.  See
    # http://stackoverflow.com/a/18418524/1382869
    def run(self):
        # Make sure the compiled Cython files in the distribution are
        # up-to-date

        try:
            import pypandoc
        except ImportError:
            raise RuntimeError(
                'Trying to create a source distribution without pypandoc. '
                'The readme will not render correctly on pypi without '
                'pypandoc so we are exiting.'
            )
        from Cython.Build import cythonize
        cythonize(ext_modules)
        _sdist.run(self)

try:
    import pypandoc
    long_description = pypandoc.convert_file('README.md', 'rst')
except (ImportError, IOError):
    with open('README.md') as file:
        long_description = file.read()

setup(name='cykdtree',
      packages=['cykdtree', 'cykdtree.tests'],
      package_dir={'cykdtree':'cykdtree'},
      package_data = {'cykdtree': ['README.md', 'README.rst'] + src_include},
      version='0.2.4',
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


