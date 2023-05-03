
from sklearn.base import ClassifierMixin

class SelfLearnClassifier(ClassifierMixin):
    def __init__(self, classifier: ClassifierMixin) -> None:
        self.classifier = classifier
    
    def fit(self, ):
        pass

    
    