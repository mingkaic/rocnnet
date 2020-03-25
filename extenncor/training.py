import os.path

import tenncor as tc

_default_cachedir = '/tmp'

_sess_prefix = 'session_'
_sess_ext = '.onnx'
_sess_nprefix = len(_sess_prefix)

_env_prefix = 'env_'
_env_ext = '.txt'

def _id_cachefile(fpath, prefix, ext):
    fname = os.path.basename(fpath)
    if os.path.isfile(fpath) and fname.startswith(prefix) and fname.endswith(ext):
        try:
            return int(fname[_sess_nprefix:_sess_nprefix + len(ext)], 16)
        except:
            print('ignoring invalid file {}'.format(fpath))
    return None

class SessCache:
    @staticmethod
    def _format_cachefile(id):
        return _sess_prefix + hex(id)[2:] + _sess_ext

    def __init__(self, cache_dir=_default_cachedir):
        self.cache_dir = cache_dir
        dirs = os.listdir(self.cache_dir)
        self.cur_id = 0
        for el in dirs:
            fileid = _id_cachefile(os.path.join(self.cache_dir, el),
                _sess_prefix, _sess_ext)
            if fileid is not None:
                self.cur_id = max(fileid)

    def store(self, session):
        cache_fpath = os.path.join(self.cache_dir,
            SessCache._format_cachefile(self.cur_id + 1))
        try:
            print('saving model "{}"'.format(cache_fpath))
            if tc.save_session_file(cache_fpath, session):
                print('successfully stored session to "{}"'.format(cache_fpath))
                self.cur_id += 1
        except Exception as e:
            print(e)
            print('failed storage to "{}"'.format(cache_fpath))

    def recover(self, session):
        cache_fpath = os.path.join(self.cache_dir,
            SessCache._format_cachefile(self.cur_id))
        try:
            print('loading model "{}'.format(cache_fpath))
            if tc.load_session_file(cache_fpath, session):
                print('successfully recovered session from "{}"'.format(cache_fpath))
        except Exception as e:
            print(e)
            print('failed recover from "{}"'.format(cache_fpath))

class EnvCache:
    def __init__(self, cache_dir=_default_cachedir):
        self.cache_dir = cache_dir
        dirs = os.listdir(self.cache_dir)
        self.cur_id = 0
        for el in dirs:
            fileid = _id_cachefile(os.path.join(self.cache_dir, el),
                _env_prefix, _env_ext)
            if fileid is not None:
                self.cur_id = max(fileid)

class CacheManager:
    _std_cachedir = _default_cachedir

    # if clean is set to True, do not recover from existing cache
    def __init__(self, name, sess, clean = False):
        self.sesscache = SessCache(os.path.join(CacheManager._std_cachedir, name))
        self.sess = sess
        if not clean:
            self.sesscache.recover(self.sess)

    def backup(self):
        self.sesscache.store(self.sess)
