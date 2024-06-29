import unittest
import pathlib
import sys

parent_dir = str(pathlib.Path(__file__).resolve().parent.parent.parent)
sys.path.append(parent_dir)


from tabularmagic._src.display.print_utils import len_ignore_format


class TestPrintWrapped(unittest.TestCase):
    def test_basic_functionality(self):
        """Ensure that print_wrapped"""
