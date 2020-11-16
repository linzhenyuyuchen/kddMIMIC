import numpy as np
from minepy import MINE
from scipy import stats
from sklearn import linear_model
from sklearn import ensemble
from sklearn import feature_selection
from itertools import permutations, combinations


#from .UF import UnionFind


"""
A two-sided P < 0.05 was used as the criterion of statistically significant
difference.

The differences in clinical characteristics between the patients in different
groups were assessed using :

Independent t test or Manne Whitney U test for continuous variables

Fisher’s exact test or chi-square test for categorical variables.

"""
class bicluster:
    def __init__(self, vec, left=None,right=None,distance=0.0,id=None) :
        self.left = left
        self.right = right
        self.vec = vec
        self.id = id
        self.distance = distance

class featureSelector():
    def __init__(self, features, labels, eps=1e-3):
        self.x = features
        self.y = labels
        self.eps = eps
        self.n = self.x.shape[1]
        self.indexs = None

    ############################
    #   univariate selection   #
    ############################
    def univarSelector(self, top_k=1, method_name="f_classif", inplace=True):
        """
        :method_name {"chi2", "f_classif", "mutual_info_classif"}
        """
        print("Feature selecting method: ", method_name)
        selector = {"chi2": feature_selection.chi2,
                    "f_classif": feature_selection.f_classif,
                    "mutual_info_classif": feature_selection.mutual_info_classif}
        func = selector[method_name]
        sler = feature_selection.SelectKBest(func, k=top_k)
        sler.fit(self.x, self.y)
        self.indexs = sler.get_support()

        if inplace:
            self.x = self.x[:, self.indexs]
            self.n = self.x.shape[1]
            return self.x, self.y
        else:
            return self.x[:, self.indexs], self.y

    #########################
    # regression selecting  #
    #########################
    def lrSelector(self, method_name="lr", inplace=True):
        """
        :method_name {"lr", "ridge"}
        """
        print("Feature selecting method: ", method_name)
        selector = {"lr": linear_model.LinearRegression(), "ridge": linear_model.Ridge()}
        lr = selector[method_name]
        lr.fit(self.x, self.y)
        coefs = lr.coef_.tolist()
        self.indexs = [i for i in range(len(coefs)) if np.abs(coefs[i]) > self.eps]
        if inplace:
            self.x = self.x[:, self.indexs]
            self.n = self.x.shape[1]
            return self.x, self.y
        else:
            return self.x[:, self.indexs], self.y

    ############################
    #      model selecting     #
    ############################
    def modelSelector(self, model_name="rf", inplace=True):
        """
        :method_name {"rf", "lasso"}
        """
        print("Feature selecting method: ", model_name)
        selector = {"rf": ensemble.RandomForestClassifier(n_estimators=10), "lasso": linear_model.LassoCV(cv=5, max_iter=5000)}
        model = selector[model_name]
        sler = feature_selection.SelectFromModel(model)
        sler.fit(self.x, self.y)
        self.indexs = sler.get_support()

        if inplace:
            self.x = self.x[:, self.indexs]
            self.n = self.x.shape[1]
            return self.x, self.y
        else:
            return self.x[:, self.indexs], self.y


    #########################
    # correlation selecting #
    #########################
    def calMic(self, x1, x2):
        # Maximal Information Coefficient
        mine = MINE(alpha=0.6, c=15)
        mine.compute_score(x1, x2)
        return mine.mic()

    def hcluster(self, X, calDistance) :
        biclusters = [ bicluster(vec = X[:, i], id = i ) for i in range(X.shape[1]) ]
        distances = {}
        flag = None
        currentclusted = -1
        print("features dim: ", len(biclusters))
        while(len(biclusters) > 1) :
            max_val = -1
            biclusters_len = len(biclusters)
            for i in range(biclusters_len-1) :
                for j in range(i + 1, biclusters_len) :
                    if distances.get((biclusters[i].id,biclusters[j].id)) == None:
                        distances[(biclusters[i].id,biclusters[j].id)], _ = calDistance(biclusters[i].vec,biclusters[j].vec)
                    d = distances[(biclusters[i].id,biclusters[j].id)] 
                    if d > max_val :
                        max_val = d
                        flag = (i,j)
            bic1,bic2 = flag
            newvec = (biclusters[bic1].vec + biclusters[bic2].vec) / 2
            newbic = bicluster(newvec, left=biclusters[bic1], right=biclusters[bic2], distance=max_val, id = currentclusted)
            currentclusted -= 1
            del biclusters[bic2]
            del biclusters[bic1]
            biclusters.append(newbic)
        return biclusters[0]

    def corrSelector(self, method_name="pearson", num=1, inplace=True):
        """
        :method_name {"pearson", "kendall", "spearman", "mic"}
        """
        print("Feature selecting method: ", method_name)
        selector = {"pearson": stats.pearsonr, "kendall": stats.kendalltau,
                    "spearman": stats.spearmanr, "mic": self.calMic}
        func = selector[method_name]
        root_node = self.hcluster(self.x, func)
        node_ids = []
        nodes = [root_node]
        while(len(nodes)>0):
            tmp = nodes.pop(0)
            if tmp.id<0:
                nodes.append(tmp.left)
                nodes.append(tmp.right)
            else:
                node_ids.append(tmp.id)
        self.indexs = np.asarray(node_ids)[:num]

        if inplace:
            self.x = self.x[:, self.indexs]
            self.n = self.x.shape[1]
            return self.x, self.y
        else:
            return self.x[:, self.indexs], self.y

    #########################
    #    T test selecting   #
    #########################
    def calTtest(self, x1, x2, threshold=0.05):
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
        stat, p = stats.levene(x1, x2)
        if p > threshold:
            _, t = stats.ttest_ind(x1, x2)
        else:
            _, t = stats.ttest_ind(x1, x2, equal_var=False)
        return t

    def ttestSelector(self, threshold=0.05, inplace=True):
        print("Feature selecting method: T test")
        labels = np.unique(self.y)
        num_labels = len(labels)
        cbs = list(combinations(range(num_labels), 2))
        num_cbs = len(cbs)
        eps = 0.25
        self.indexs = []
        for i in range(self.n):
            count = 0
            for c in range(num_cbs):
                index1, index2 = cbs[c]
                row1, row2 = labels[index1], labels[index2]
                r1, r2 = self.y==row1, self.y==row2
                t = self.calTtest(self.x[r1, i], self.x[r2, i])
                if t > threshold:
                    count += 1
            if count/num_cbs < eps:
                self.indexs.append(i)

        if inplace:
            self.x = self.x[:, self.indexs]
            self.n = self.x.shape[1]
            return self.x, self.y
        else:
            return self.x[:, self.indexs], self.y

    #########################
    #   mann whitney u test #
    #########################
    def mannSelector(self, threshold=0.05, inplace=True):
        print("Feature selecting method: mann whitney u test")
        labels = np.unique(self.y)
        num_labels = len(labels)
        cbs = list(combinations(range(num_labels), 2))
        num_cbs = len(cbs)
        eps = 0.25
        self.indexs = []
        for i in range(self.n):
            count = 0
            for c in range(num_cbs):
                index1, index2 = cbs[c]
                row1, row2 = labels[index1], labels[index2]
                r1, r2 = self.y==row1, self.y==row2
                _, t = stats.mannwhitneyu(self.x[r1, i], self.x[r2, i], alternative='two-sided')
                if t > threshold:
                    count += 1
            if count/num_cbs < eps:
                self.indexs.append(i)

        if inplace:
            self.x = self.x[:, self.indexs]
            self.n = self.x.shape[1]
            return self.x, self.y
        else:
            return self.x[:, self.indexs], self.y



    # https://scikit-learn.org/stable/modules/feature_selection.html

