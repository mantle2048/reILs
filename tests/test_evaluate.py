# +
from reILs.infrastructure.execution.evaluate import create_parser, run, evaluate
from reILs.user_config import LOCAL_DIR

# %reload_ext autoreload
# %autoreload 2
# %matplotlib notebook
# -

parser = create_parser()
args = parser.parse_args([
    '--exp-dir',
    'data/gail_HalfCheetah-v2/gail_HalfCheetah-v2_5',
    '--track-progress',
    '--render'
])
print(args)
run(args)








