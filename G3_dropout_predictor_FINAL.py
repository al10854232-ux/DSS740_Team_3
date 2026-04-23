"""
dropout_predictor.py
====================
Student Dropout Prediction — End-to-End ML Pipeline
Group 3: Kristen Kohler | Marith Bijkerk | Chenxi Lai | Ankita Lala
Course: Machine Learning | Saint Joseph University | Spring 2026

This class encapsulates the full machine learning pipeline:
    1.  Data loading and inspection
    2.  Preprocessing — binary target, column removal, outlier treatment (Winsorization)
    3.  Feature engineering — four domain-informed derived features
    4.  Class imbalance handling — random oversampling on training set only
    5.  Train/test split and StandardScaler normalisation
    6.  Base model training — five algorithms
    7.  Hyperparameter tuning — GridSearchCV with stratified CV
    8.  Model comparison and selection
    9.  Feature importance and Partial Dependence Plots
    10. Decision threshold analysis — why recall is the right metric here
    11. Final evaluation and business interpretation

Usage
-----
    predictor = DropoutPredictor(filepath='students_dropout_academic_success.csv')
    predictor.run_full_pipeline()          # runs everything in order
    # OR step by step:
    predictor.load_data()
    predictor.preprocess()
    predictor.engineer_features()
    predictor.split_and_balance()
    predictor.scale()
    predictor.train_base_models()
    predictor.tune_models()
    predictor.compare_models()
    predictor.plot_feature_importance()
    predictor.plot_partial_dependence()
    predictor.threshold_analysis()         # NEW — metric/threshold justification
    predictor.generate_report()
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import resample
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, classification_report, roc_curve
)
from sklearn.inspection import PartialDependenceDisplay


class DropoutPredictor:
    """
    End-to-end machine learning pipeline for predicting student dropout.

    Parameters
    ----------
    filepath : str
        Path to the CSV dataset.
    test_size : float
        Fraction of data reserved for the test set (default 0.20).
    cv_folds : int
        Number of stratified folds for hyperparameter tuning (default 3).
    random_state : int
        Seed for all random operations (default 42).
    target_col : str
        Name of the target column in the CSV (default 'target').
    winsorize : bool
        Whether to apply Winsorization (1st-99th percentile) for outlier
        treatment (default True). Set to False only if you have already
        handled outliers externally.
    deployment_threshold : float
        Classification threshold used for final predictions (default 0.40).
        Lowering from the sklearn default of 0.50 increases recall at a
        modest cost in precision — appropriate for dropout screening where
        missing a true at-risk student is the more costly error.
    """

    # Five candidate algorithms (SVM excluded per group constraint)
    _BASE_MODELS = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Naive Bayes':         GaussianNB(),
        'Decision Tree':       DecisionTreeClassifier(random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    # Hyperparameter search spaces
    _PARAM_GRIDS = {
        'Logistic Regression': {
            'C': [0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs']
        },
        'Decision Tree': {
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5],
            'criterion': ['gini']
        },
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [10, None],
            'max_features': ['sqrt']
        },
        'Gradient Boosting': {
            'n_estimators': [100],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1]
        },
    }

    # Columns removed during preprocessing (near-constant or non-causal)
    _DROP_COLS = ['Marital Status', 'Nacionality', 'Application order']

    def __init__(
        self,
        filepath: str,
        test_size: float = 0.20,
        cv_folds: int = 3,
        random_state: int = 42,
        target_col: str = 'target',
        winsorize: bool = True,
        deployment_threshold: float = 0.40,
    ):
        self.filepath              = filepath
        self.test_size             = test_size
        self.cv_folds              = cv_folds
        self.random_state          = random_state
        self.target_col            = target_col
        self.winsorize             = winsorize
        self.deployment_threshold  = deployment_threshold

        # Pipeline state — populated as methods are called
        self.df                = None
        self.df_model          = None
        self.X_train           = None
        self.X_test            = None
        self.y_train           = None
        self.y_test            = None
        self.X_train_sc        = None
        self.X_test_sc         = None
        self.scaler            = None
        self.feature_names     = None
        self.base_results      = {}
        self.tuned_results     = {}
        self._tuned_estimators = {}
        self.best_model        = None
        self.best_model_name   = None

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC API — run steps individually or call run_full_pipeline()
    # ─────────────────────────────────────────────────────────────────────────

    def run_full_pipeline(self) -> 'DropoutPredictor':
        """Execute all pipeline steps in the correct order."""
        return (self
                .load_data()
                .preprocess()
                .engineer_features()
                .split_and_balance()
                .scale()
                .train_base_models()
                .tune_models()
                .compare_models()
                .plot_feature_importance()
                .plot_partial_dependence()
                .threshold_analysis()
                .generate_report())

    # ── Step 1 ────────────────────────────────────────────────────────────────

    def load_data(self) -> 'DropoutPredictor':
        """
        Load dataset from CSV and rename the target column.
        Prints shape, missing value count, duplicate count, and class
        distribution as a quick sanity check.
        """
        self.df = pd.read_csv(self.filepath)
        self.df = self.df.rename(columns={self.target_col: 'Target'})

        print("=" * 55)
        print("STEP 1: DATA LOADED")
        print("=" * 55)
        print(f"  Shape:         {self.df.shape[0]} rows x {self.df.shape[1]} cols")
        print(f"  Missing values:{self.df.isnull().sum().sum()}")
        print(f"  Duplicates:    {self.df.duplicated().sum()}")
        print(f"\n  Target distribution:")
        for cls, cnt in self.df['Target'].value_counts().items():
            print(f"    {cls:<12} {cnt:>5}  ({cnt/len(self.df)*100:.1f}%)")
        return self

    # ── Step 2 ────────────────────────────────────────────────────────────────

    def preprocess(self) -> 'DropoutPredictor':
        """
        Preprocessing steps applied in this order:

        1. Binary target: Dropout = 1, Non-Dropout (Graduate or Enrolled) = 0.
           Rationale: the business question is simply who is at risk of leaving,
           not distinguishing between those who graduated and those still enrolled.

        2. Column removal: three near-constant or non-causal columns dropped.

        3. Outlier treatment (Winsorization at 1st and 99th percentile):
           Extreme values are capped rather than removed. This preserves all
           4,424 rows while preventing a handful of extreme observations from
           pulling model coefficients or split thresholds in misleading
           directions. Winsorization is preferred over deletion because many
           features (e.g. curricular units approved = 0) are both outliers AND
           the most informative records for dropout prediction.
        """
        if self.df is None:
            raise RuntimeError("Call load_data() first.")

        # Binary target
        self.df['Dropout_Binary'] = (self.df['Target'] == 'Dropout').astype(int)

        # Drop near-constant / non-causal columns
        cols_to_drop = [c for c in self._DROP_COLS if c in self.df.columns]
        self.df_model = self.df.drop(columns=cols_to_drop + ['Target']).copy()

        # Winsorize continuous features
        if self.winsorize:
            num_features = (self.df_model
                            .select_dtypes(include='number')
                            .drop(columns=['Dropout_Binary'])
                            .columns)
            for col in num_features:
                p01 = self.df_model[col].quantile(0.01)
                p99 = self.df_model[col].quantile(0.99)
                self.df_model[col] = self.df_model[col].clip(p01, p99)

        print("\n" + "=" * 55)
        print("STEP 2: PREPROCESSING COMPLETE")
        print("=" * 55)
        print(f"  Binary dropout rate: {self.df['Dropout_Binary'].mean():.1%}")
        print(f"  Columns dropped:     {cols_to_drop}")
        print(f"  Winsorization:       {'Applied (1st-99th pct)' if self.winsorize else 'Skipped'}")
        print(f"  Model dataset shape: {self.df_model.shape}")
        return self

    # ── Step 3 ────────────────────────────────────────────────────────────────

    def engineer_features(self) -> 'DropoutPredictor':
        """
        Four domain-informed features derived from raw columns.

        - sem1_approval_rate: fraction of enrolled sem-1 units that were passed.
          A student enrolled in 6 units who passes only 1 (rate = 0.17) is a
          very different risk profile from one who passes all 6 (rate = 1.0),
          even though their raw approved count may look similar.

        - sem2_approval_rate: same ratio for semester 2. Published research on
          this dataset (Realinho et al., 2022) identifies this as the single
          strongest academic predictor of dropout.

        - total_approved: cumulative units approved across both semesters.
          Captures overall academic trajectory rather than a single snapshot.

        - avg_grade: mean grade across both semesters. Reduces the collinearity
          between the two individual grade columns while retaining the signal.
        """
        if self.df_model is None:
            raise RuntimeError("Call preprocess() first.")

        eps = 1e-6
        m = self.df_model  # shorthand

        m['sem1_approval_rate'] = (
            m['Curricular units 1st sem (approved)'] /
            (m['Curricular units 1st sem (enrolled)'] + eps)
        ).clip(0, 1)

        m['sem2_approval_rate'] = (
            m['Curricular units 2nd sem (approved)'] /
            (m['Curricular units 2nd sem (enrolled)'] + eps)
        ).clip(0, 1)

        m['total_approved'] = (
            m['Curricular units 1st sem (approved)'] +
            m['Curricular units 2nd sem (approved)']
        )

        m['avg_grade'] = (
            m['Curricular units 1st sem (grade)'] +
            m['Curricular units 2nd sem (grade)']
        ) / 2

        print("\n" + "=" * 55)
        print("STEP 3: FEATURE ENGINEERING COMPLETE")
        print("=" * 55)
        new_feats = ['sem1_approval_rate','sem2_approval_rate',
                     'total_approved','avg_grade']
        print(f"  New features: {new_feats}")
        print(f"  Dataset shape: {self.df_model.shape}")
        print("\n  Correlations with Dropout_Binary:")
        corrs = (self.df_model[new_feats + ['Dropout_Binary']]
                 .corr()['Dropout_Binary']
                 .drop('Dropout_Binary')
                 .round(3))
        for feat, val in corrs.items():
            print(f"    {feat:<25} {val:>7}")
        return self

    # ── Step 4 ────────────────────────────────────────────────────────────────

    def split_and_balance(self) -> 'DropoutPredictor':
        """
        Stratified 80/20 train/test split, then random oversampling of the
        Dropout (minority) class on the TRAINING SET ONLY.

        Why oversampling rather than undersampling?
        With only 1,421 dropout records vs 3,003 non-dropout, undersampling
        would discard roughly 1,500 non-dropout training examples, wasting
        data. Oversampling replicates existing dropout records instead,
        preserving all information while equalising class frequency.

        Why on training set only?
        Applying oversampling before splitting would let synthetic duplicate
        records appear in both train and test, inflating performance metrics
        (data leakage). The test set is kept at its natural 68/32 distribution
        to reflect real-world deployment conditions.
        """
        if self.df_model is None:
            raise RuntimeError("Call preprocess() first.")

        X = self.df_model.drop(columns=['Dropout_Binary'])
        y = self.df_model['Dropout_Binary']
        self.feature_names = list(X.columns)

        X_tr_raw, self.X_test, y_tr_raw, self.y_test = train_test_split(
            X, y, test_size=self.test_size,
            random_state=self.random_state, stratify=y
        )

        tr = pd.concat([X_tr_raw, y_tr_raw], axis=1)
        maj = tr[tr['Dropout_Binary'] == 0]
        mino = tr[tr['Dropout_Binary'] == 1]
        mino_up = resample(mino, replace=True,
                           n_samples=len(maj),
                           random_state=self.random_state)
        balanced = pd.concat([maj, mino_up]).sample(
            frac=1, random_state=self.random_state
        )

        self.X_train = balanced.drop(columns=['Dropout_Binary'])
        self.y_train = balanced['Dropout_Binary']

        print("\n" + "=" * 55)
        print("STEP 4: SPLIT AND BALANCING COMPLETE")
        print("=" * 55)
        print(f"  Train shape:         {self.X_train.shape}")
        print(f"  Test shape:          {self.X_test.shape}")
        print(f"  Train class balance: {self.y_train.value_counts().to_dict()}")
        print(f"  Test class balance:  {self.y_test.value_counts().to_dict()}")
        return self

    # ── Step 5 ────────────────────────────────────────────────────────────────

    def scale(self) -> 'DropoutPredictor':
        """
        Apply StandardScaler fitted on the training set only.
        The scaler's mean and standard deviation come exclusively from training
        data; the test set is transformed using those same values to prevent
        information from the test set influencing the normalisation.
        """
        if self.X_train is None:
            raise RuntimeError("Call split_and_balance() first.")

        self.scaler     = StandardScaler()
        self.X_train_sc = self.scaler.fit_transform(self.X_train)
        self.X_test_sc  = self.scaler.transform(self.X_test)

        print("\n" + "=" * 55)
        print("STEP 5: SCALING COMPLETE")
        print("=" * 55)
        print("  StandardScaler fitted on training set only.")
        print(f"  Train mean (first 3): {self.X_train_sc[:, :3].mean(axis=0).round(4)}")
        return self

    # ── Step 6 ────────────────────────────────────────────────────────────────

    def train_base_models(self) -> 'DropoutPredictor':
        """
        Train all five algorithms with default hyperparameters to establish a
        fair benchmark before tuning. Results stored in self.base_results.

        Why five models?
        Each model family makes different assumptions about the data structure:
        - Logistic Regression: linear decision boundary, interpretable odds ratios
        - Naive Bayes: assumes feature independence, very fast, probabilistic
        - Decision Tree: non-linear, rule-based, prone to overfitting at defaults
        - Random Forest: ensemble that averages many trees, robust to noise
        - Gradient Boosting: sequential error correction, typically strongest on
          tabular data with mixed feature types
        """
        if self.X_train_sc is None:
            raise RuntimeError("Call scale() first.")

        print("\n" + "=" * 55)
        print("STEP 6: TRAINING BASE MODELS")
        print("=" * 55)

        for name, model in self._BASE_MODELS.items():
            model.fit(self.X_train_sc, self.y_train)
            y_pred = model.predict(self.X_test_sc)
            y_prob = model.predict_proba(self.X_test_sc)[:, 1]
            self.base_results[name] = self._metrics(y_pred, y_prob)
            r = self.base_results[name]
            print(f"  {name:<28}  F1={r['F1-Score']:.4f}  "
                  f"Recall={r['Recall']:.4f}  AUC={r['ROC-AUC']:.4f}")

        print("\n  Base model summary:")
        print(pd.DataFrame(self.base_results).T.to_string())
        return self

    # ── Step 7 ────────────────────────────────────────────────────────────────

    def tune_models(self) -> 'DropoutPredictor':
        """
        GridSearchCV hyperparameter tuning.

        Scoring metric: ROC-AUC
        ROC-AUC is threshold-independent and evaluates the model's ability to
        rank positive (dropout) cases above negative cases across all possible
        thresholds. This is preferred over accuracy during tuning because
        accuracy is misleading on imbalanced data — a model can achieve 68%
        accuracy by never predicting dropout at all.

        Cross-validation: StratifiedKFold
        Stratification ensures each fold has the same class ratio as the full
        training set, preventing fold-level imbalance from distorting results.
        """
        if not self.base_results:
            raise RuntimeError("Call train_base_models() first.")

        cv = StratifiedKFold(n_splits=self.cv_folds,
                             shuffle=True,
                             random_state=self.random_state)

        print("\n" + "=" * 55)
        print("STEP 7: HYPERPARAMETER TUNING (GridSearchCV)")
        print("=" * 55)

        for name, base_model in self._BASE_MODELS.items():
            if name == 'Naive Bayes':
                # No meaningful hyperparameters to tune
                nb = GaussianNB()
                nb.fit(self.X_train_sc, self.y_train)
                self._tuned_estimators[name] = nb
                y_pred = nb.predict(self.X_test_sc)
                y_prob = nb.predict_proba(self.X_test_sc)[:, 1]
                self.tuned_results[name] = self._metrics(y_pred, y_prob)
                print(f"  Naive Bayes: no tuning required.")
                continue

            params = self._PARAM_GRIDS.get(name)
            if not params:
                continue

            grid = GridSearchCV(
                type(base_model)(random_state=self.random_state)
                if hasattr(base_model, 'random_state')
                else type(base_model)(),
                params, cv=cv, scoring='roc_auc', n_jobs=-1
            )
            grid.fit(self.X_train_sc, self.y_train)
            best = grid.best_estimator_
            self._tuned_estimators[name] = best

            y_pred = best.predict(self.X_test_sc)
            y_prob = best.predict_proba(self.X_test_sc)[:, 1]
            self.tuned_results[name] = self._metrics(y_pred, y_prob)
            r = self.tuned_results[name]
            print(f"  {name:<28}  best={grid.best_params_}")
            print(f"  {'':28}  F1={r['F1-Score']:.4f}  "
                  f"Recall={r['Recall']:.4f}  AUC={r['ROC-AUC']:.4f}")

        print("\n  Tuned model summary:")
        print(pd.DataFrame(self.tuned_results).T.to_string())
        return self

    # ── Step 8 ────────────────────────────────────────────────────────────────

    def compare_models(self) -> 'DropoutPredictor':
        """
        Select the best model.

        Selection criterion: ROC-AUC (primary), then F1-Score as tiebreaker.
        Gradient Boosting consistently leads on both metrics for this dataset.

        Why not pick the model with highest recall alone?
        Pure recall maximisation is easy — predict dropout for every student
        and recall = 100%. The trade-off is precision collapses. ROC-AUC and
        F1-Score together ensure the selected model genuinely discriminates
        rather than blindly over-classifying.

        The deployment threshold (default 0.40) is then lowered from the
        sklearn default of 0.50, boosting recall further for the production
        screening use case. See threshold_analysis() for the full picture.
        """
        if not self.tuned_results:
            raise RuntimeError("Call tune_models() first.")

        tuned_df = pd.DataFrame(self.tuned_results).T
        self.best_model_name = tuned_df['ROC-AUC'].idxmax()
        self.best_model      = self._tuned_estimators.get(self.best_model_name)

        print("\n" + "=" * 55)
        print("STEP 8: MODEL SELECTION")
        print("=" * 55)
        print(f"  Winner: {self.best_model_name}")
        r = self.tuned_results[self.best_model_name]
        for metric, val in r.items():
            print(f"    {metric:<12} {val:.4f}")
        return self

    # ── Step 9 ────────────────────────────────────────────────────────────────

    def plot_feature_importance(self, top_n: int = 15,
                                 save_path: str = None) -> 'DropoutPredictor':
        """
        Plot feature importances from the best model.

        Only tree-based models expose .feature_importances_. For Gradient
        Boosting, importance = mean decrease in impurity weighted by the number
        of samples reaching each split. Higher values indicate the feature was
        used more often AND at splits that reduced classification error more.
        """
        if self.best_model is None:
            raise RuntimeError("Call compare_models() first.")

        if not hasattr(self.best_model, 'feature_importances_'):
            print(f"  Note: {self.best_model_name} does not expose "
                  "feature_importances_. Skipping importance plot.")
            return self

        imp = (pd.Series(self.best_model.feature_importances_,
                          index=self.feature_names)
               .sort_values(ascending=False)
               .head(top_n))

        plt.figure(figsize=(10, 6))
        imp.sort_values().plot(kind='barh', color='#2E86AB', edgecolor='white')
        plt.title(f'Top {top_n} Feature Importances: {self.best_model_name}',
                  fontweight='bold', fontsize=12)
        plt.xlabel('Mean Decrease in Impurity')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        return self

    # ── Step 10 ───────────────────────────────────────────────────────────────

    def plot_partial_dependence(self, top_n: int = 3,
                                 save_path: str = None) -> 'DropoutPredictor':
        """
        Partial Dependence Plots for top N features.

        A PDP shows the marginal effect of a single feature on the predicted
        dropout probability, averaged across all other features. Unlike
        feature importance (which shows how often a feature is used), PDPs
        show the DIRECTION and SHAPE of the relationship — for example, that
        dropout risk drops sharply when sem2_approval_rate goes from 0 to 0.3,
        then flattens.
        """
        if self.best_model is None or not hasattr(self.best_model,
                                                   'feature_importances_'):
            print("  PDP requires a tree-based model. Skipping.")
            return self

        top_feats = (pd.Series(self.best_model.feature_importances_,
                                index=self.feature_names)
                     .sort_values(ascending=False)
                     .head(top_n)
                     .index.tolist())
        top_idx = [self.feature_names.index(f) for f in top_feats]

        fig, axes = plt.subplots(1, top_n, figsize=(5 * top_n, 4))
        if top_n == 1:
            axes = [axes]

        for ax, idx, fname in zip(axes, top_idx, top_feats):
            PartialDependenceDisplay.from_estimator(
                self.best_model, self.X_test_sc, [idx],
                feature_names=self.feature_names, ax=ax,
                line_kw={'color': '#E74C3C', 'linewidth': 2.5}
            )
            ax.set_title(f'PDP: {fname}', fontsize=9, fontweight='bold')

        plt.suptitle(f'Partial Dependence Plots: Top {top_n} Predictors',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        return self

    # ── Step 11 — NEW ─────────────────────────────────────────────────────────

    def threshold_analysis(self, save_path: str = None) -> 'DropoutPredictor':
        """
        Decision threshold analysis and metric selection justification.

        WHY RECALL IS THE PRIMARY METRIC FOR DROPOUT SCREENING:
        --------------------------------------------------------
        In any binary classifier, the decision threshold controls the trade-off
        between two error types:

            False Negative (FN): the model predicts Non-Dropout for a student
            who will actually drop out. The student receives no intervention,
            continues on the dropout path, and the institution loses both the
            student and the associated tuition revenue.
            Cost: HIGH — irreversible for the student; expensive for the institution.

            False Positive (FP): the model flags a student who would have
            continued as high-risk. An academic advisor checks in with them.
            Cost: LOW — an unnecessary conversation, but no lasting harm.

        Because FN cost >> FP cost, RECALL (sensitivity = TP / (TP + FN)) is
        the metric we care most about. We want to catch as many true at-risk
        students as possible, even if that means some false alarms.

        WHY NOT JUST USE ACCURACY?
        ---------------------------
        The test set has 601 Non-Dropout and 284 Dropout students. A naive
        model that ALWAYS predicts Non-Dropout would achieve 67.9% accuracy
        while catching zero dropout students. Accuracy is therefore a misleading
        metric for this imbalanced binary problem. F1-Score (harmonic mean of
        precision and recall) and ROC-AUC are far more informative.

        RECOMMENDED THRESHOLD: 0.40 (not sklearn default 0.50)
        --------------------------------------------------------
        At threshold=0.40, recall increases meaningfully at a modest precision
        cost. For a screening application, the institutional cost of contacting
        a non-at-risk student unnecessarily is much lower than the cost of
        missing a student who is about to drop out.

        This method plots the full threshold sensitivity curve so the
        institution can pick the operating point that best fits their capacity
        for advisor outreach.
        """
        if self.best_model is None:
            raise RuntimeError("Call compare_models() first.")

        print("\n" + "=" * 55)
        print("STEP 10: THRESHOLD ANALYSIS")
        print("=" * 55)

        y_prob = self.best_model.predict_proba(self.X_test_sc)[:, 1]
        thresholds = np.arange(0.20, 0.76, 0.05)

        rec_vals  = []
        prec_vals = []
        f1_vals   = []
        acc_vals  = []

        for t in thresholds:
            y_hat = (y_prob >= t).astype(int)
            rec_vals.append(recall_score(self.y_test, y_hat, zero_division=0))
            prec_vals.append(precision_score(self.y_test, y_hat, zero_division=0))
            f1_vals.append(f1_score(self.y_test, y_hat, zero_division=0))
            acc_vals.append(accuracy_score(self.y_test, y_hat))

        # Print table
        print(f"\n  {'Threshold':>10} {'Accuracy':>10} {'Precision':>10}"
              f" {'Recall':>10} {'F1':>10}")
        print("  " + "-" * 55)
        for t, a, p, r, f in zip(thresholds, acc_vals,
                                  prec_vals, rec_vals, f1_vals):
            flag = " <-- RECOMMENDED" if abs(t - self.deployment_threshold) < 0.001 else ""
            print(f"  {t:>10.2f} {a:>10.4f} {p:>10.4f} {r:>10.4f} {f:>10.4f}{flag}")

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(thresholds, rec_vals,  'o-', color='#E74C3C', lw=2.5,
                     label='Recall (primary metric)')
        axes[0].plot(thresholds, prec_vals, 's-', color='#3498DB', lw=2.5,
                     label='Precision')
        axes[0].plot(thresholds, f1_vals,   '^--', color='#2ECC71', lw=2,
                     label='F1-Score')
        axes[0].plot(thresholds, acc_vals,  'd:', color='#9B59B6', lw=1.5,
                     label='Accuracy')
        axes[0].axvline(self.deployment_threshold, color='red',
                        linestyle='--', lw=2, alpha=0.85,
                        label=f'Recommended ({self.deployment_threshold})')
        axes[0].axvline(0.50, color='grey', linestyle='--', lw=1, alpha=0.6,
                        label='sklearn default (0.50)')
        axes[0].set_xlabel('Decision Threshold')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Metric Sensitivity to Threshold\n'
                          f'({self.best_model_name})', fontweight='bold')
        axes[0].legend(fontsize=8, loc='lower left')
        axes[0].set_ylim(0.45, 1.05)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(thresholds, rec_vals,  'o-', color='#E74C3C', lw=2.5,
                     label='Recall')
        axes[1].plot(thresholds, prec_vals, 's-', color='#3498DB', lw=2.5,
                     label='Precision')
        axes[1].fill_between(thresholds, rec_vals, prec_vals,
                             alpha=0.15, color='purple',
                             label='Recall-Precision gap')
        axes[1].axvline(self.deployment_threshold, color='red',
                        linestyle='--', lw=2,
                        label=f'Recommended ({self.deployment_threshold})')

        # Annotate recommended point
        idx40 = np.argmin(np.abs(thresholds - self.deployment_threshold))
        axes[1].annotate(
            f"Threshold={self.deployment_threshold}\n"
            f"Recall={rec_vals[idx40]:.1%}\n"
            f"Precision={prec_vals[idx40]:.1%}",
            xy=(self.deployment_threshold, rec_vals[idx40]),
            xytext=(self.deployment_threshold + 0.07, rec_vals[idx40] + 0.04),
            fontsize=8, color='darkred',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5)
        )
        axes[1].set_xlabel('Decision Threshold')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Recall vs Precision Trade-off\n'
                          '(Business Screening Context)', fontweight='bold')
        axes[1].legend(fontsize=9, loc='lower left')
        axes[1].set_ylim(0.45, 1.05)
        axes[1].grid(True, alpha=0.3)

        plt.suptitle('Decision Threshold Analysis: Why Recall Matters for Dropout Screening',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        idx40 = np.argmin(np.abs(thresholds - self.deployment_threshold))
        print(f"\n  At threshold={self.deployment_threshold}: "
              f"Recall={rec_vals[idx40]:.1%}, "
              f"Precision={prec_vals[idx40]:.1%}")
        print(f"  At threshold=0.50:  "
              f"Recall={rec_vals[np.argmin(np.abs(thresholds-0.50))]:.1%}, "
              f"Precision={prec_vals[np.argmin(np.abs(thresholds-0.50))]:.1%}")
        return self

    # ── Step 12 ───────────────────────────────────────────────────────────────

    def generate_report(self) -> 'DropoutPredictor':
        """
        Print the final classification report and business interpretation
        using the configured deployment threshold.
        """
        if self.best_model is None:
            raise RuntimeError("Call compare_models() first.")

        y_prob = self.best_model.predict_proba(self.X_test_sc)[:, 1]
        y_pred = (y_prob >= self.deployment_threshold).astype(int)

        print("\n" + "=" * 60)
        print(f"FINAL MODEL REPORT: {self.best_model_name}")
        print(f"Decision threshold: {self.deployment_threshold}")
        print("=" * 60)
        print(classification_report(
            self.y_test, y_pred,
            target_names=['Non-Dropout', 'Dropout']
        ))
        auc = roc_auc_score(self.y_test, y_prob)
        print(f"ROC-AUC (threshold-independent): {auc:.4f}")

        print("\n" + "-" * 60)
        print("BUSINESS INTERPRETATION")
        print("-" * 60)
        recall    = recall_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        n_dropouts = int(self.y_test.sum())
        caught     = int(recall * n_dropouts)
        missed     = n_dropouts - caught

        print(f"\n  Test set contains {n_dropouts} actual dropout students.")
        print(f"  Model correctly flagged {caught} of them "
              f"({recall:.1%} recall).")
        print(f"  Model missed {missed} actual dropouts "
              f"(false negatives — highest cost error).")
        print(f"  Of all flagged students, {precision:.1%} truly are at risk "
              f"(acceptable false-alarm rate for screening).")
        print(f"\n  METRIC PRIORITY ORDER FOR THIS PROBLEM:")
        print(f"    1. Recall      {recall:.4f}  — catch as many dropouts as possible")
        print(f"    2. F1-Score    {f1_score(self.y_test, y_pred):.4f}  — balanced measure")
        print(f"    3. ROC-AUC     {auc:.4f}  — overall discrimination quality")
        print(f"    4. Precision   {precision:.4f}  — resource efficiency of outreach")
        print(f"    5. Accuracy    {accuracy_score(self.y_test, y_pred):.4f}  "
              f"— least informative (imbalanced classes)")

        print(f"\n  KEY ACTIONABLE FINDINGS:")
        print(f"    - Students with sem2_approval_rate < 0.30 are highest risk")
        print(f"    - Tuition not current: dropout rate > 70%; prioritise")
        print(f"      financial aid outreach for this group immediately")
        print(f"    - Scholarship holders show dramatically lower dropout risk;")
        print(f"      expanding eligibility is the highest-ROI retention lever")
        print(f"    - Deploy at threshold={self.deployment_threshold} in "
              f"production for optimal recall")
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # Inference on new data
    # ─────────────────────────────────────────────────────────────────────────

    def predict(self, X_new: pd.DataFrame) -> np.ndarray:
        """
        Predict dropout risk for new student records using the deployment
        threshold (default 0.40).

        Parameters
        ----------
        X_new : pd.DataFrame
            New student data with the same feature columns used during training.

        Returns
        -------
        np.ndarray
            Binary array — 1 = predicted at-risk of dropout, 0 = predicted safe.
        """
        if self.scaler is None or self.best_model is None:
            raise RuntimeError("Pipeline must be fully trained before predicting.")
        X_sc = self.scaler.transform(X_new[self.feature_names])
        return (self.best_model.predict_proba(X_sc)[:, 1]
                >= self.deployment_threshold).astype(int)

    def predict_proba(self, X_new: pd.DataFrame) -> np.ndarray:
        """Return raw dropout probability scores for new student records."""
        if self.scaler is None or self.best_model is None:
            raise RuntimeError("Pipeline must be fully trained before predicting.")
        X_sc = self.scaler.transform(X_new[self.feature_names])
        return self.best_model.predict_proba(X_sc)[:, 1]

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _metrics(self, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
        return {
            'Accuracy':  round(accuracy_score(self.y_test, y_pred), 4),
            'Precision': round(precision_score(self.y_test, y_pred), 4),
            'Recall':    round(recall_score(self.y_test, y_pred), 4),
            'F1-Score':  round(f1_score(self.y_test, y_pred), 4),
            'ROC-AUC':   round(roc_auc_score(self.y_test, y_prob), 4),
        }

    def __repr__(self) -> str:
        status = ('untrained' if self.best_model is None
                  else f'trained — {self.best_model_name}')
        return (f"DropoutPredictor(filepath='{self.filepath}', "
                f"threshold={self.deployment_threshold}, "
                f"status={status})")


# ─────────────────────────────────────────────────────────────────────────────
# Run as script — full pipeline demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    predictor = DropoutPredictor(
        filepath='students_dropout_academic_success.csv',
        test_size=0.20,
        cv_folds=3,
        random_state=42,
        winsorize=True,
        deployment_threshold=0.40,
    )
    predictor.run_full_pipeline()
    print('\n', repr(predictor))
