


class PyEquivTracker():

    def __init__(self):
        self._preprocessing_code = ''

    def add_to_preprocessing_code(self, code: str):
        self._preprocessing_code += '\n' + code
        
    def get_preprocessing_code(self):
        return self._preprocessing_code







