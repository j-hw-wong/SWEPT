# Configuration file for the Sphinx documentation builder.
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../../angular_binning'))
sys.path.insert(0, os.path.abspath('../../gaussian_cl_likelihood'))
sys.path.insert(0, os.path.abspath('../../catalogue_sim'))
sys.path.insert(0, os.path.abspath('../../pcl_measurement'))
sys.path.insert(0, os.path.abspath('../../inference_analysis'))

autodoc_mock_imports = ['gaussian_cl_likelihood', 'angular_binning', 'healpy', 'matplotlib', 'numpy', 'pymaster', 'scipy', 'h5py', 'pyFlask']


# -- Project information

project = 'SWEPT'
copyright = '2025'
author = 'Jonathan Wong'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'myst_parser'
]

source_suffix = ['.rst', '.md']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

from sphinx.application import Sphinx
from sphinx.util.docfields import Field


def setup(app: Sphinx):
    app.add_object_type(
        'confval',
        'confval',
        objname='configuration value',
        indextemplate='pair: %s; configuration value',
        doc_field_types=[
            Field('type', label='Type', has_arg=False, names=('type',)),
            Field('default', label='Default', has_arg=False, names=('default',)),
        ]
    )