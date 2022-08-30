import os
import os.path as osp
cur_dir = os.getcwd()
par_dir = os.path.dirname(cur_dir)
LOCAL_LOG_DIR= osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))), 'data')
LOCAL_EXP_DIR= osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))), 'exp')
LOCAL_DIR = osp.abspath(osp.dirname(osp.dirname(__file__)))
LOCAL_RENDER_CONFIG = {
    "mode": 'rgb_array',
    "width": 500,
    "height": 500,
    "camera_id": None,
    "camera_name": None,
}
LOCAL_MP4_CONFIG = {
    "fps": 10,
    "backend": 'MoviePy', # 'PIL', 'MoviePy'
    "scale": 1.0,
}
