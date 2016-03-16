import os.path

class DirType(object):
    def __call__(self, arg):
        assert arg is None or isinstance(arg, str) or isinstance(arg, unicode)
        if arg is None:
            return None
        elif not arg:
            return '.'
        else:
            assert os.path.isdir(arg), \
                'Path %s does not exist or is not a directory' % arg
            return os.path.normpath(arg)
