import sys
from pipeline import *

infile = sys.argv[1]
outfile = sys.argv[2]

infile = 'example/' + infile
outfile = 'example/' + outfile

process_obj2obja(infile, outfile)