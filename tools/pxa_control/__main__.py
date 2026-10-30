"""`python -m pxa_control`: the desktop entry (pxa_control.desktop.main)."""
import sys

from .desktop import main

sys.exit(main())
