import pandas as pd
from pgmpy.estimators import HillClimbSearch, BicScore, BayesianEstimator
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from pgmpy.estimators import MaximumLikelihoodEstimator
df = pd.read_csv("2020_bn_nb_data.txt", sep=r"\s+")
course_columns = df.columns[:-1]
hc = HillClimbSearch(df[course_columns])
best_model = hc.estimate(scoring_method=BicScore(df[course_columns]))
model = BayesianModel(best_model.edges())
model.fit(df[course_columns], estimator=BayesianEstimator, prior_type="BDeu")
print("\nLearned structure:\n", best_model.edges())
for node in model.nodes():
    print(f"\n{node} CPT:\n", model.get_cpds(node))
evidence = {'EC100': 'DD', 'IT101': 'CC', 'MA101': 'CD'}
evidence = {k: v for k, v in evidence.items() if k in model.nodes()}
infer = VariableElimination(model)
q = infer.query(variables=['PH100'], evidence=evidence, show_progress=False)
print("\nP(PH100 | EC100=DD, IT101=CC, MA101=CD):")
print(q)
print("\n" + "-"*60 + "\n")
for col in df.columns:
    df[col] = df[col].astype("category")
feature_cols = df.columns[:-1]
target_col = df.columns[-1]
n_repeats = 20
nb_accuracies = []
X_full = df[feature_cols]
y_full = df[target_col]
for _ in range(n_repeats):
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_full, y_full, train_size=0.7, stratify=y_full, random_state=None
    )
    X_train_enc = pd.DataFrame(index=X_train_raw.index)
    X_test_enc = pd.DataFrame(index=X_test_raw.index)
    for col in feature_cols:
        train_values = X_train_raw[col].astype(str).unique().tolist()
        try:
            mode_val = X_train_raw[col].mode().iloc[0]
        except IndexError:
            mode_val = train_values[0] if train_values else ""
        X_test_col = X_test_raw[col].astype(str).copy()
        unseen_mask = ~X_test_col.isin(train_values)
        if unseen_mask.any():
            X_test_col.loc[unseen_mask] = str(mode_val)
        X_train_col = X_train_raw[col].astype(str).copy()
        mapping = {val: i for i, val in enumerate(train_values)}
        X_train_enc[col] = X_train_col.map(mapping).astype(int)
        X_test_enc[col] = X_test_col.map(mapping).astype(int)
    target_values = y_train_raw.astype(str).unique().tolist()
    target_mapping = {val: i for i, val in enumerate(target_values)}
    if not target_values:
        raise ValueError("No classes found in training target.")
    y_test_series = y_test_raw.astype(str).copy()
    y_test_series[~y_test_series.isin(target_values)] = str(y_train_raw.mode().iloc[0])
    y_train_enc = y_train_raw.astype(str).map(target_mapping).astype(int)
    y_test_enc = y_test_series.map(target_mapping).astype(int)
    clf = CategoricalNB()
    clf.fit(X_train_enc, y_train_enc)
    acc = clf.score(X_test_enc, y_test_enc)
    nb_accuracies.append(acc)
print(f"Naive Bayes (independent) {n_repeats}x mean accuracy: {np.mean(nb_accuracies)*100:.2f}%")
print(f"Std deviation: {np.std(nb_accuracies)*100:.2f}%")
print("\n" + "-"*60 + "\n")
lf = pd.read_csv("2020_bn_nb_data.txt", sep=r"\s+")
course_columns = lf.columns[:-1]
target_col = lf.columns[-1]
bn_accuracies = []
n_repeats = 20
for _ in range(n_repeats):
    train, test = train_test_split(lf, train_size=0.7, stratify=lf[target_col])
    hc = HillClimbSearch(train[course_columns])
    model_edges = hc.estimate(scoring_method=BicScore(train[course_columns])).edges()
    all_edges = list(model_edges) + [(c, target_col) for c in course_columns]
    model = BayesianModel(all_edges)
    model.fit(train, estimator=MaximumLikelihoodEstimator)

    infer = VariableElimination(model)
    correct = 0
    label_map = {cat: code for code, cat in enumerate(train[target_col].astype('category').cat.categories)}
    for _, row in test.iterrows():
        evidence = row[course_columns].to_dict()
        q = infer.map_query(variables=[target_col], evidence=evidence)
        pred_label = q[target_col]
        actual_label = row[target_col]
        correct += (pred_label == label_map[actual_label])
    acc = correct / len(test)
    bn_accuracies.append(acc)
print(f"BayesianNet (dependent) {n_repeats}x mean accuracy: {np.mean(bn_accuracies)*100:.2f}%")
print(f"Std deviation: {np.std(bn_accuracies)*100:.2f}%")
