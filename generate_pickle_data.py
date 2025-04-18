import subprocess
import time
import pickle as pickle
import sys
import logging

# Setup logging
LOG_FILENAME = 'lifespan2.log'
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)

sample_size = 5

for sample in range(sample_size):
    fname = "Adex_" + str(sample + 1) + ".p"
    cmd = ["python3", "Adex_lifespan_prediction.py", fname]  
    try:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=sys.stdout, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if err:
            logging.error(time.strftime('%a %H:%M:%S') + ":: Error in sample " + str(sample) + " -> " + err.decode('utf-8'))
        else:
            logging.debug(time.strftime('%a %H:%M:%S') + ":: Done with sample " + str(sample))
    except Exception as e:
        logging.error(time.strftime('%a %H:%M:%S') + ":: Exception occurred -> " + str(e))
   