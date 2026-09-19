"""
The six competing classifiers, each wrapped in a leak-free sklearn Pipeline
that accepts raw epoch arrays X of shape (n_trials, n_channels, n_times).

  svm        : CSP -> StandardScaler -> SVC(RBF)
  rf         : CSP -> RandomForest
  fbcsp      : Filter Banks CSP -> SVC(RBF)
  mdm        : Covariances -> MDM
  tangent    : Covariences -> Tangent Space -> Logistic Regression
  fb tangent : Filter Banks -> Tangent Space -> Ridge

All six expose the standard fit / predict / predict_proba-ish API so the
training and the online code treat them identically.
"""

from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM

from mne.decoding import CSP
import mne

from . import config as C

# --------------------------------------------------------------------------- #
# Custom Transformer for Filter Bank CSP classifiers
# --------------------------------------------------------------------------- #
class FilterBankCSP(BaseEstimator, TransformerMixin):
    """
    Divides the signal to smaller overlapping frequency bands in order to
    extract features from each of them, utilizing CSP algorithm. 
    """
    def __init__(self, sfreq=128, n_components=4):
        self.sfreq = sfreq
        self.n_components = n_components
        self.bands = [
            (2, 4), (4, 8), (6, 10), (8, 12), (10, 14), (12, 16), (14, 18), 
            (16, 20), (18, 22), (20, 24), (22, 26), (24, 30)
        ]
        self.csps = [CSP(n_components=self.n_components, reg="ledoit_wolf", log=True, norm_trace=False) 
                     for _ in self.bands]

    def _filter_band(self, X, l_freq, h_freq):
        # Filtering the 3D array (n_trials, n_channels, n_times)
        return mne.filter.filter_data(
            X, sfreq=self.sfreq, l_freq=l_freq, h_freq=h_freq, 
            verbose="ERROR", fir_design="firwin"
        )

    def fit(self, X, y):
        for i, (l_freq, h_freq) in enumerate(self.bands):
            X_band = self._filter_band(X, l_freq, h_freq)
            self.csps[i].fit(X_band, y)
        return self

    def transform(self, X):
        features = []
        for i, (l_freq, h_freq) in enumerate(self.bands):
            X_band = self._filter_band(X, l_freq, h_freq)
            feat = self.csps[i].transform(X_band)
            features.append(feat)
        return np.concatenate(features, axis=1)

# Custom Transformer for Filter Bank Tangent Space classifier
class FilterBankTangentSpace(BaseEstimator, TransformerMixin):
    def __init__(self, sfreq=128):
        self.sfreq = sfreq
        self.bands = [
            (2, 6), (4,8), (6,10), (8, 12), (10, 14), (12, 16), (14, 18), 
            (16, 20), (18, 22), (20, 24), (22, 26), (24, 30)
        ]
        
        self.covs = [Covariances(estimator="lwf") for _ in self.bands]
        self.tangents = [TangentSpace(metric="riemann") for _ in self.bands]

    def _filter_band(self, X, l_freq, h_freq):
        return mne.filter.filter_data(
            X, sfreq=self.sfreq, l_freq=l_freq, h_freq=h_freq, 
            verbose="ERROR", fir_design="firwin"
        )

    def fit(self, X, y):
        for i, (l_freq, h_freq) in enumerate(self.bands):
            X_band = self._filter_band(X, l_freq, h_freq)
            cov_matrices = self.covs[i].fit_transform(X_band, y)
            self.tangents[i].fit(cov_matrices, y)
        return self

    def transform(self, X):
        features = []
        for i, (l_freq, h_freq) in enumerate(self.bands):
            X_band = self._filter_band(X, l_freq, h_freq)
            cov_matrices = self.covs[i].transform(X_band)
            tangent_feats = self.tangents[i].transform(cov_matrices)
            features.append(tangent_feats)
        return np.concatenate(features, axis=1)


def build_models() -> dict[str, Pipeline]:
    """Return {name: unfitted sklearn Pipeline}."""
    # CSP with log-variance output  as the classic MI feature extractor.
    def make_csp():
        return CSP(n_components=C.N_CSP_COMPONENTS, reg="ledoit_wolf",
                   log=True, norm_trace=False)

    models: dict[str, Pipeline] = {}

    # 1) SVM on CSP features
    models["svm"] = Pipeline([
        ("csp", make_csp()),
        ("scale", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1.0, gamma="scale",
                    class_weight="balanced", probability=True,
                    random_state=C.RANDOM_STATE)),
    ])

    # 2) Random Forest on CSP features
    models["rf"] = Pipeline([
        ("csp", make_csp()),
        ("clf", RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=C.RANDOM_STATE, n_jobs=-1)),
    ])

    # 3) FBCSP (Filter Bank CSP) + Mutual Info Feature K-Best Selection + SVM
    models["fbcsp"] = Pipeline([
        ("fb_csp", FilterBankCSP(sfreq=C.SFREQ_TARGET, n_components=4)),
        ("select", SelectKBest(score_func=mutual_info_classif, k=20)),
        ("pca", PCA(n_components=0.95, random_state=C.RANDOM_STATE)),
        #("scale", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=C.RANDOM_STATE)),
    ])

    # 4) Riemannian MDM
    models["mdm"] = Pipeline([
        ("cov", Covariances(estimator="lwf")),
        ("clf", MDM(metric="riemann")),
    ])

    # 5) Riemannian Tangent Space + Logistic Regression
    models["tangent_space"] = Pipeline([
        ("cov", Covariances(estimator="lwf")),
        ("tangent", TangentSpace(metric="riemann")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, class_weight="balanced", random_state=C.RANDOM_STATE)),
    ])

    # 6) Filter Bank Tanget + Mutual Info Feature K-Best Selection + PCA + any simple classifier (Logistic Regression / Ridge)
    models["fb_tangent"] = Pipeline([
        ("fb_tangent", FilterBankTangentSpace(sfreq=C.SFREQ_TARGET)),
        ("scale", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=50)),
        ("pca", PCA(n_components=0.95, random_state=C.RANDOM_STATE)),
        #("select", SelectKBest(score_func=f_classif, k=50)),
        # ("clf", LogisticRegression(C=0.1, penalty="l2", class_weight="balanced", random_state=C.RANDOM_STATE)),
        ("clf", RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), class_weight="balanced")),
    ])

    return models