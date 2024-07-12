import tabularmagic as tm


class _SharedAnalyzerWrapper:
    """A wrapper for a shared analyzer."""

    def __init__(self):
        self.shared_analyzer = None

    def set_shared_analyzer(self, shared_analyzer: tm.Analyzer):
        self.shared_analyzer = shared_analyzer

    def get_shared_analyzer(self) -> tm.Analyzer:
        return self.shared_analyzer


shared_analyzer = _SharedAnalyzerWrapper()
