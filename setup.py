from setuptools import setup
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Compiler.Options import directive_defaults
import numpy
import os

release = False

RTDFLAG = bool(os.environ.get('READTHEDOCS', None) == 'True')

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

# Needed for line_profiler - disable for production code
if not release:
    directive_defaults['profile'] = True
    directive_defaults['linetrace'] = True
    directive_defaults['binding'] = True

cmdclass = { }
ext_modules = [ ]

ext_options = dict(language="c++",
                   include_dirs=[numpy.get_include()],
                   extra_compile_args=["-std=c++11"])  # "-std=gnu++11")
# CYTHON_TRACE required for coverage and line_profiler.  Remove for release.
if not release:
    ext_options['define_macros'] = [('CYTHON_TRACE', '1')]

if RTDFLAG:
    ext_options['libraries'] = []
    ext_options['extra_link_args'] = []
    ext_options['extra_compile_args'].append('-DREADTHEDOCS')

def make_cpp(cpp_file):
    if not os.path.isfile(cpp_file):
        open(cpp_file,'a').close()
        assert(os.path.isfile(cpp_file))

if use_cython:
    make_cpp("cykdtree/c_kdtree.cpp")
    make_cpp("cykdtree/c_utils.cpp")
    make_cpp("cykdtree/c_parallel_kdtree.cpp")
    ext_modules += cythonize(Extension("cykdtree/kdtree",
                                       sources=["cykdtree/kdtree.pyx",
                                                "cykdtree/c_kdtree.cpp",
                                                "cykdtree/c_utils.cpp"],
                                       language="c++",
                                       include_dirs=[numpy.get_include()],
                                       extra_compile_args=["-std=gnu++11"]))
    ext_modules += cythonize(Extension("cykdtree/utils",
                                       sources=["cykdtree/utils.pyx",
                                                "cykdtree/c_utils.cpp"],
                                       language="c++",
                                       include_dirs=[numpy.get_include()],
                                       extra_compile_args=["-std=gnu++11"]))
    ext_modules += cythonize(Extension("cykdtree/parallel_kdtree",
                                       sources=["cykdtree/parallel_kdtree.pyx",
                                                "cykdtree/c_parallel_kdtree.cpp",
                                                "cykdtree/c_kdtree.cpp"],
                                       language="c++",
                                       include_dirs=[numpy.get_include()],
                                       extra_compile_args=["-std=gnu++11"]))
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules += [
        Extension("cykdtree.kdtree", ["cykdtree/c_kdtree.cpp"],
                  include_dirs=[numpy.get_include()]),
        Extension("cykdtree.utils", ["cykdtree/c_utils.cpp"],
                  include_dirs=[numpy.get_include()]),
        Extension("cykdtree.parallel_kdtree", ["cykdtree/c_parallel_kdtree.cpp"],
                  include_dirs=[numpy.get_include()]),
    ]

with open('README.rst') as file:
    long_description = file.read()

setup(name='cykdtree',
      packages=['cykdtree'],
      package_dir={'cykdtree':'cykdtree'},
      package_data = {'cykdtree': ['README.md', 'README.rst'],
      version='0.1',
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
      # package_data = {'':['*.pxd']},
      zip_safe=False,
      cmdclass = cmdclass,
      ext_modules = ext_modules)


