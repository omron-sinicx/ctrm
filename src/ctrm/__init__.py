"""logger config
Author: Keisuke Okumura
Affiliation: TokyoTech
"""

import json
import logging
import logging.config
import os

# set logger
logconf = json.loads(
    open(
        os.path.join(os.path.dirname(__file__), "logconf.json"),
        encoding="UTF-8",
    ).read()
)
logging.config.dictConfig(logconf)


__version__ = "0.1.0"
