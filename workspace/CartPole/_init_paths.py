""" setup python path """
# import os
import sys
# import json


# def add_environ(cfg):
#     with open(cfg) as f:
#         environ = json.loads(f.read())
#         for k in environ:
#             os.environ[k] = environ[k]
#             print("Set os.environ: `%s`" % k)


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


# add_environ('../ENVIRON')

add_path('../../')
print("add code root path (with `rllib`).")
