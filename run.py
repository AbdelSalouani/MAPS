import sys
from chunk_7_10_complete_pipeline import MAPSProgressiveEncoder

infile = sys.argv[1]
outfile = sys.argv[2]

infile = 'obja/example/' + infile
outfile = 'obja/example/' + outfile

encoder = MAPSProgressiveEncoder()
encoder.process_obj_to_obja(infile, outfile)

