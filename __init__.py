# -*- coding: utf-8 -*-
###############################################################################
# ██████╗██████╗██╗  ██╗██████╗      ██████╗  ██████╗  ██████╗ ██╗    ██████╗ #
# ╚═██╔═╝██╔═══╝╚██╗██╔╝╚═██╔═╝      ╚═██╔═╝ ██╔═══██╗██╔═══██╗██║    ██╔═══╝ #
#   ██║  █████╗  ╚███╔╝   ██║          ██║   ██║   ██║██║   ██║██║    ██████╗ #
#   ██║  ██╔══╝  ██╔██╗   ██║          ██║   ██║   ██║██║   ██║██║    ╚═══██║ #
#   ██║  ██████╗██╔╝ ██╗  ██║ ███████╗ ██║   ╚██████╔╝╚██████╔╝██████╗██████║ #
#   ╚═╝  ╚═════╝╚═╝  ╚═╝  ╚═╝ ╚══════╝ ╚═╝    ╚═════╝  ╚═════╝ ╚═════╝╚═════╝ #
###############################################################################
# A python package designed for text including:
# 1. parsing, preprocessing, and representation of text data
# 2. entity resolution and temporal resolution
# 3. general statistics from text

from . import extraction
from . import preprocessing
from . import readability
from . import resolution
from . import similarity
from . import stat
from . import tokens
from . import vectorizer
from . import vocab_tools
from . import words
from . import highlighting

from . import ml_training
from . import ml_visualizations
