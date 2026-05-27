import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

# Note for end user: Use only forward_selection() function


class FeatureSelector:
    def __init__(
        self,
        model,
        score_function,
        top_ns=(200, 300, 400),
        test_size=0.3,
        random_state=None,
        verbose=True,
        features = None,
        forward = True
    ):
        self.model = model
        self.score_function = score_function
        self.top_ns = top_ns
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose
        self.features = features
        self.forward = forward

    def _evaluate_subset(self, features, X_train, X_test, y_train, y_test):
        model = clone(self.model)
        model.fit(X_train[:, features], y_train)

        y_proba = model.predict_proba(X_test[:, features])[:, 1]

        order = np.argsort(-y_proba)

        scores = []
        opt_tresh = []
        for top_n in self.top_ns:
            idx = order[:top_n]

            y_true_top = y_test[idx]
            max_scores = []
            threshs = [0.2, 0.04, 0.05, 0.06, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            for thresh in threshs:
                y_pred_top = (y_proba[idx] > thresh).astype(int)

                score = self.score_function(y_true_top, y_pred_top, len(features))
                max_scores.append(score)
            scores.append(max(max_scores))
            opt_tresh.append(threshs[np.argmax(max_scores)])

        return np.max(scores), self.top_ns[np.argmax(scores)], opt_tresh[np.argmax(scores)]

    def fit(self, X_train, y_train, X_test, y_test):
        X_train = np.array(X_train)
        y_train = np.array(y_train).ravel()

        X_test = np.array(X_test)
        y_test = np.array(y_test).ravel()

        n_features = X_train.shape[1]

        if self.forward == True:
            remaining_features = self.features
            selected_features = []

            best_score = -np.inf

            while len(selected_features) < 50:
                scores = []

                for feature in remaining_features:
                    candidate_features = selected_features + [feature]

                    score, best_subset, best_thresh  = self._evaluate_subset(
                        candidate_features, X_train, X_test, y_train, y_test
                    )

                    scores.append((feature, score, best_subset, best_thresh))

                best_feature, best_candidate_score, best_candidates, best_threshold = max(scores, key=lambda x: x[1])

                if best_candidate_score < best_score:
                    break

                selected_features.append(best_feature)
                remaining_features.remove(best_feature)
                best_score = best_candidate_score
                global_best_candidates = best_candidates
                global_best_threshold = best_threshold


        elif self.forward == False:
            selected_features = self.features.copy() 

            
            best_score, best_candidates, best_threshold = self._evaluate_subset(
                selected_features, X_train, X_test, y_train, y_test
            )


            while len(selected_features) > 0:
                scores = []

                for feature in selected_features:
                    candidate_features = [f for f in selected_features if f != feature]

                    score, best_subset, best_thresh = self._evaluate_subset(
                        candidate_features, X_train, X_test, y_train, y_test
                    )

                    scores.append((feature, score, best_subset, best_thresh))

                worst_feature, best_candidate_score, best_candidates, best_threshold = max(
                    scores, key=lambda x: x[1]
                )
                
                if best_candidate_score < best_score:
                    break
                
                selected_features.remove(worst_feature)
                best_score = best_candidate_score
                global_best_candidates = best_candidates
                global_best_threshold = best_threshold


        self.selected_features_ = selected_features
        self.best_score_ = best_score
        self.best_threshold = global_best_threshold
        self.best_candidates = global_best_candidates
            

        if self.verbose:
            print(f"Features Selected: {self.selected_features_} \nScore Function: {self.best_score_}\nThreshold: {self.best_threshold}\nCandidates Number: {self.best_candidates}")

        return self

    def transform(self, X):
        return X[:, self.selected_features_]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
    
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split

def single_feature_ranking(
    X_train,
    y_train,
    X_test,
    y_test,
    model,
    score_function,
    verbose=False
    ):
    X_train = np.array(X_train)
    y_train = np.array(y_train).ravel()

    X_test = np.array(X_test)
    y_test = np.array(y_test).ravel()

    scores = []
    acs = []
    bas = []

    for j in range(X_train.shape[1]):
        model_ = clone(model)

        model_.fit(X_train[:, [j]], y_train)

        y_proba = model_.predict_proba(X_test[:, [j]])[:, 1]

        y_true_top = y_test

        s =[]
        a=[]
        b=[]
        opt_tresh=[]
        thresholds=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for thres in thresholds:
            y_pred_top = (y_proba > thres).astype(int)

            score = score_function(y_true_top, y_pred_top, 1)
            accuracy = accuracy_score(y_true_top, y_pred_top)
            ba = balanced_accuracy_score(y_true_top, y_pred_top)
            s.append(score)
            a.append(accuracy)
            b.append(ba)

        scores.append(max(s))
        acs.append(a[np.argmax(s)])
        bas.append(b[np.argmax(s)])
        opt_tresh.append(thresholds[np.argmax(s)])
        if verbose:
            print(f"Feature {j}, best score: {max(s)}, thresh: {opt_tresh}")
            
    ranking = np.argsort(scores)[::-1]

    return ranking, np.array(scores), np.array(acs), np.array(bas)

def score_function(y_true, y_pred, n_features):
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tp = np.sum((y_pred == 1) & (y_true == 1))

    cost = 10*tp - 5*fp - 200*n_features
    return cost


def forward_selection(X_train,y_train, X_test, y_test, model=GradientBoostingClassifier(n_estimators=10)):
    ranking, _, _, _ = single_feature_ranking(X_train,y_train, X_test, y_test, model, score_function, verbose=False)
    selector = FeatureSelector(
    model=model,
    score_function=score_function,
    top_ns=[obs_no for obs_no in range(5, 1005, 5)],
    features=list(ranking[:50]),
    verbose=True
    )

    selector.fit(X_train, y_train, X_test, y_test)

    return selector.selected_features_, selector.best_threshold, selector.best_candidates