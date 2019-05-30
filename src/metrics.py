import time
import numpy as np

SHOW_PLOT = False


class ClassifierMetrics():
    scores = []
    fit_times = []
    test_times = []
    name = ""
    clf = None
    
    def __init__(self, clf=None, name=""):
        self.name = name
        self.clf = clf
        self.scores = []
        self.fit_times = []
        self.test_times = []
        self.facets_count = []
    
    def fit_and_measure(self,trainX, trainY,testX, testY):
        start = time.clock()
        # testX = preprocessing.normalize(testX, norm='max', axis=0)
        # testX = pw.rbf_kernel(testX)
        self.clf.fit(trainX, trainY)
        
        self.add_fit_time(time.clock() - start)
        
        start = time.clock()
        if( not("NCHC" in self.name)):
            self.add_score(self.clf.score(testX, testY))
        else:
            self.add_score(self.clf.score(testX, testY))
            self.add_facets_count(self.clf.get_total_facets())
            if(SHOW_PLOT):
                # self.clf.plot_all_convexs(testX=testX, testY=testY)
                predicts = self.clf.predict(testX)
                probs = self.clf.predict_proba(testX)
                for x,y,p,prob in zip(testX, testY, predicts, probs):
                    if(y!=p):
                        print(self.clf.distances(x), prob)
                        self.clf.plot_all_convexs(testX=np.array([x]), testY=np.array([y]))
        
        self.add_test_time(time.clock() - start)

    def add_score(self,value):
        self.scores.append(value)
    def add_facets_count(self,value):
        self.facets_count.append(value)
    def add_fit_time(self,value):
        self.fit_times.append(value)
    def add_test_time(self,value):
        self.test_times.append(value)
    
    def print_scores(self):
        print("Scores:",self.scores)
    def print_fit_times(self):
        print("Fit times:",self.fit_times)
    def print_test_times(self):
        print("Test times:",self.test_times)
    def print_facets_count(self):
        print("Facets count:",self.facets_count)
    
    

    def print_metrics(self, mean=False):
        if(not mean):
            self.print_scores()
            self.print_fit_times()
            self.print_test_times()
            self.print_facets_count()
        else:
            str_score_mean = np.mean(self.scores)
            str_fit_time_mean = np.mean(self.fit_times)
            str_score_time_mean = np.mean(self.test_times)
            
            if("NCHC" in self.name):
                str_facets_count_mean = np.mean(self.facets_count)
                str_facets_count_std = np.std(self.facets_count)

                str_to_print = "{:>25}  {:>25}  {:>25}  {:>25}  {:>25} {:>25}".format(
                    self.name,
                    str_score_mean,
                    str_fit_time_mean,
                    str_score_time_mean,
                    str_facets_count_mean,
                    str_facets_count_std
                )
                str_to_print+="\nFaces:"+str(self.facets_count)
            else:
                str_to_print = "{:>25}  {:>25}  {:>25}  {:>25}".format(
                    self.name,
                    str_score_mean,
                    str_fit_time_mean,
                    str_score_time_mean
                )

            print(str_to_print)

class ClassifierMetricsSet():
    metrics = {}
    
    def __init__(self,clfs=[],names=[]):
        for clf, name in zip(clfs, names):
            self.metrics[name] = ClassifierMetrics(clf, name)
        
    def add_clf(self,clf,name):
        self.metrics[name] = ClassifierMetrics(clf, name)

    def get_clf(self,name):
        return self.metrics[name]

    def print_metrics(self, mean=False):
        str_to_print = "{:>25}  {:>25}  {:>25}  {:>25}".format(
            "Classifier Name",
            "Score Mean",
            "Fit time mean",
            "Test time mean"
        )
        for clf in self.metrics:
            self.metrics[clf].print_metrics(mean)
