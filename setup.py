from setuptools import setup
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Compiler.Options import directive_defaults
import numpy
import os

RTDFLAG = bool(os.environ.get('READTHEDOCS', None) == 'True')

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True


# Needed for line_profiler - disable for production code
directive_defaults['linetrace'] = True
directive_defaults['binding'] = True

cmdclass = { }
ext_modules = [ ]

ext_options = dict(language="c++",
                       include_dirs=[numpy.get_include()],
                       extra_compile_args=["-std=c++11"],# "-std=gnu++11",
                       # CYTHON_TRACE required for coverage and line_profiler.  Remove for release.
                       define_macros=[('CYTHON_TRACE', '1')])
if RTDFLAG:
    ext_options['libraries'] = []
    ext_options['extra_link_args'] = []
    ext_options['extra_compile_args'].append('-DREADTHEDOCS')

if use_cython:
    ext_modules += cythonize(Extension("cykdtree/kdtree",
                                       sources=["cykdtree/kdtree.pyx","cykdtree/c_kdtree.cpp","cykdtree/c_utils.cpp"],
                                       language="c++",
                                       include_dirs=[numpy.get_include()],
                                       extra_compile_args=["-std=gnu++11"]))
    ext_modules += cythonize(Extension("cykdtree/utils",
                                       sources=["cykdtree/utils.pyx","cykdtree/c_utils.cpp"],
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
    ]

setup(name='cykdtree',
      version='0.1',
      description='Python interface for CGAL Triangulations',
      url='https://langmm@bitbucket.org/langmm/cykdtree',
      author='Meagan Lang',
      author_email='langmm.astro@gmail.com',
      license='GPL',
      packages=['cykdtree'],
      package_data = {'':['*.pxd']},
      zip_safe=False,
      cmdclass = cmdclass,
      ext_modules = ext_modules)


