import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')
import functools
print = functools.partial(print, flush=True)
import os
os.makedirs("results, exist_ok=True")

# Import pymoo for NSGA-II
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

# Import TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def create_model(input_dim, learning_rate):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy')
    return model

class NeuralNetworkProblem(Problem):
    def __init__(self, X_train, y_train, X_test, y_test, sensitive_train, sensitive_test):
        # Decision variables:
        # [learning_rate, threshold, fairness_weight, explainability_weight, epoch]
        # learning_rate in [1e-5, 1e-1], threshold in [0, 1],
        # fairness_weight and explainability_weight in [0,1],
        # epoch in [5, 50] (will be rounded to nearest integer)
        super().__init__(n_var=5, n_obj=3, n_constr=0, 
                         xl=np.array([1e-5, 0.0, 0.0, 0.0, 5]),
                         xu=np.array([1e-1, 1.0, 1.0, 1.0, 50]))
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.sensitive_train = sensitive_train
        self.sensitive_test = sensitive_test
        
        # Parameters for GESD calculation.
        self.n_perturbations = 5
        self.perturbation_scale = 0.1
        
    def _evaluate(self, x, out, *args, **kwargs):
        n_points = x.shape[0]
        f = np.zeros((n_points, 3))
        
        for i in range(n_points):
            learning_rate = x[i, 0]
            threshold = x[i, 1]
            fairness_weight = x[i, 2]
            explainability_weight = x[i, 3]
            epoch = int(round(x[i, 4]))
            
            try:
                # Create and train the neural network with 'epoch' epochs.
                model = create_model(self.X_train.shape[1], learning_rate)
                model.fit(self.X_train, self.y_train, epochs=epoch, batch_size=32, verbose=0)
                
                y_pred_prob = model.predict(self.X_test).flatten()
                y_pred = (y_pred_prob >= threshold).astype(int)
                
                auc = roc_auc_score(self.y_test, y_pred_prob)
                dp_diff = demographic_parity_difference(
                    self.y_test, 
                    y_pred, 
                    sensitive_features=self.sensitive_test
                )
                gesd, _ = self.calculate_gesd(model, self.X_test, self.sensitive_test, self.X_train)
                
                # Combine objectives: We weight the explainability and fairness metrics.
                f[i, 0] = -auc
                f[i, 1] =  gesd
                f[i, 2] =  dp_diff
            except Exception as e:
                print(f"Error: {e}")
                f[i, 0] = -0.5
                f[i, 1] = 1.0
                f[i, 2] = 1.0
                
        out["F"] = f

    def calculate_gesd(self, model, X, sensitive_attr, X_train, 
                            num_perturbations=10, 
                            gaussian_scale=0.1, 
                            mask_prob=0.2, 
                            baseline_value=0.0,
                            sample_size=100):
        def get_aggregated_explanation(instance, n_features, model, shap_explainer, lime_explainer):
            # Compute SHAP explanation using DeepExplainer for neural networks.
            shap_vals = shap_explainer.shap_values(instance.reshape(1, -1))
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]
            shap_vector = np.squeeze(shap_vals)
            if shap_vector.ndim !=1:
                shap_vector=shap_vector.flatten()

            # Compute LIME explanation.
            def predict_fn(x):
                probs = model.predict(x)
                two_class_probs = np.zeros((x.shape[0], 2))
                two_class_probs[:, 0] = 1-probs.flatten()
                two_class_probs[:, 1] = probs.flatten()
                return two_class_probs

            lime_exp = lime_explainer.explain_instance(
                data_row=instance,
                predict_fn=predict_fn,
                num_features=n_features,
                top_labels = 1
            )

            lime_map = lime_exp.as_map()
            if  lime_map:
                key = list(lime_map.keys())[0]
                lime_values_list = lime_map[key]
            else:
                lime_values_list = []
            lime_vector = np.zeros(n_features)
            for feat_idx, weight in lime_values_list:
                lime_vector[feat_idx] = weight
    
            agg_explanation = (shap_vector + lime_vector) / 2.0
            return agg_explanation
        
        n_features = X.shape[1]
        background_indices = np.random.choice(X_train.shape[0], min(100, X_train.shape[0]), replace=False)
        background = X_train[background_indices]
        shap_explainer = shap.DeepExplainer(model, background)
        lime_explainer = LimeTabularExplainer(
            training_data=X_train,
            feature_names=None,
            class_names=['class0', 'class1'],
            mode='classification'
        )
    
        sample_size = min(sample_size, X.shape[0])
        sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
        X_sample = X[sample_indices]
        sensitive_sample = sensitive_attr[sample_indices]
    
        stability_scores = []
        groups_list = []
    
        for i, instance in enumerate(X_sample):
            original_exp = get_aggregated_explanation(instance, n_features, model, shap_explainer, lime_explainer)
            instab_scores = []
            for _ in range(num_perturbations):
                perturbed = instance + np.random.normal(0, gaussian_scale, size=instance.shape)
                mask = np.random.rand(n_features) < mask_prob
                perturbed[mask] = baseline_value
                perturbed_exp = get_aggregated_explanation(perturbed, n_features, model, shap_explainer, lime_explainer)
                distance = np.linalg.norm(original_exp - perturbed_exp, ord=1)
                stability = 1 / (1 + distance)
                instab_scores.append(stability)
            avg_stability = np.mean(instab_scores)
            stability_scores.append(avg_stability)
            groups_list.append(sensitive_sample[i])
    
        stability_scores = np.array(stability_scores)
        groups_list = np.array(groups_list)
        unique_groups = np.unique(groups_list)
        group_stabilities = {group: np.mean(stability_scores[groups_list == group]) for group in unique_groups}
    
        if len(unique_groups) == 2:
            robust_gesd = np.abs(group_stabilities[unique_groups[0]] - group_stabilities[unique_groups[1]])
        else:
            overall_stability = np.mean(list(group_stabilities.values()))
            robust_gesd = np.mean([(group_stabilities[group] - overall_stability)**2 for group in unique_groups])
    
        return robust_gesd, group_stabilities

def main():
    data = pd.read_csv("bail.csv")
    categorical_features = data.select_dtypes(include=['object', 'bool'])
    for col in categorical_features:
        le = LabelEncoder()
        data[col] = data[col].fillna('missing')
        data[col] = le.fit_transform(data[col])

    X = data.drop(['MALE', 'RECID'], axis=1).values
    y= data['RECID'].values
    sensitive_attr = data['WHITE'].values

    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive_attr, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create problem instance.
    problem = NeuralNetworkProblem(X_train, y_train, X_test, y_test, sens_train, sens_test)
    
    # Configure NSGA-II algorithm.
    algorithm = NSGA2(
        pop_size=30,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    
    print("Starting optimization...")
    termination = MaximumGenerationTermination(15)
    results = minimize(problem, algorithm, termination, verbose=True, seed=42)
    
    Xs = results.X
    Fs = results.F
    Fs[:, 0] = -Fs[:, 0]  # Convert AUC back to positive.
    
    print("\nPareto optimal solutions:")
    for i in range(len(Xs)):
        print(f"Solution {i+1}:")
        print(f"  Parameters: learning_rate={Xs[i, 0]:.6f}, threshold={Xs[i, 1]:.4f}, fairness_weight={Xs[i, 2]:.4f}, explainability_weight={Xs[i, 3]:.4f}, epoch={int(round(Xs[i, 4]))}")
        print(f"  Metrics: AUC={Fs[i, 0]:.4f}, GESD={Fs[i, 1]:.4f}, DP = {Fs[i, 2]:.4f}")
    
    # Visualize Pareto front.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(Fs[:, 0], Fs[:, 1], Fs[:, 2], s=80, alpha=0.8, c=Fs[:, 2], cmap='viridis')
    cbar = plt.colorbar(sc)
    cbar.set_label('Demographic Parity Difference')
    ax.set_xlabel('AUC (maximize)')
    ax.set_ylabel('GESD (Minimize)')
    ax.set_zlabel('DP Difference (minimize)')
    ax.set_title('Pareto Front of NSGA-II Optimization for Recidivism Dataset')
    plt.tight_layout()
    plt.savefig('pareto_front_pymoo_nn_epoch_bail.png')
    
    utilities = []
    for i in range(len(Fs)):
        norm_auc = Fs[i, 0]
        max_gesd = np.max(Fs[:, 1])
        min_gesd = np.min(Fs[:, 1])
        norm_gesd = 1 - ((Fs[i, 1] - min_gesd) / (max_gesd - min_gesd)) if (max_gesd - min_gesd) > 0 else 1.0
        max_dp = np.max(Fs[:, 2])
        min_dp = np.min(Fs[:, 2])
        norm_dp = 1 - ((Fs[i, 2] - min_dp) / (max_dp - min_dp)) if (max_dp - min_dp) > 0 else 1.0
        
        fairness_weight = Xs[i, 2]
        explainability_weight = Xs[i, 3]
        utility =  norm_auc + explainability_weight * norm_gesd + fairness_weight * norm_dp
        utilities.append((i, utility))
    
    best_idx = max(utilities, key=lambda x: x[1])[0]
    best_solution = Xs[best_idx]
    
    print("\nRecommended balanced solution:")
    print(f"  Parameters: learning_rate={best_solution[0]:.6f}, threshold={best_solution[1]:.4f}, fairness_weight={best_solution[2]:.4f}, explainability_weight={best_solution[3]:.4f}, epoch={int(round(best_solution[4]))}")
    print(f"  Metrics: AUC={Fs[best_idx, 0]:.4f}, GESD={Fs[best_idx, 1]:.4f}, DP Diff={Fs[best_idx, 2]:.4f}")
    
    final_model = create_model(X_train.shape[1], best_solution[0])
    final_epoch = int(round(best_solution[4]))
    final_model.fit(X_train, y_train, epochs=final_epoch, batch_size=32, verbose=0)
    y_pred_prob = final_model.predict(X_test).flatten()
    y_pred = (y_pred_prob >= best_solution[1]).astype(int)
    
    final_auc = roc_auc_score(y_test, y_pred_prob)
    final_gesd, group_stabilities = problem.calculate_gesd(final_model, X_test, sens_test, X_train)
    final_dp = demographic_parity_difference(y_test, y_pred, sensitive_features=sens_test)
    final_eod = equalized_odds_difference(y_test, y_pred, sensitive_features=sens_test)
    final_f1 = f1_score(y_test, y_pred)
    
    with open("results/metrics_bail.txt", "w") as f:
        f.write(f"AUC: {final_auc}\n")
        f.write(f"DP: {final_dp}\n")
        f.write(f"GESD: {final_gesd}\n")
        f.write(f"EOD: {final_eod}\n")
        f.write(f"f1: {final_f1}\n")
    
    print("\nFinal model detailed evaluation:")
    print(f"  AUC: {final_auc:.4f}")
    print(f"  GESD: {final_gesd:.4f}")
    print(f"  Demographic Parity Difference: {final_dp:.4f}")

    
    def analyze_explanations_by_group(model, X, sensitive):
        # Use the first 100 samples for explanation
        X_subset = X[:100]
    
        # Create a SHAP explainer
        shap_explainer = shap.Explainer(model, X_subset)
        # Create a LIME explainer
        lime_explainer = LimeTabularExplainer(
            training_data=X_subset,
            feature_names=[f"F{i}" for i in range(X.shape[1])],
            class_names=['class0', 'class1'],
            mode='classification'
        )
    
        groups = np.unique(sensitive)
        group_shap_values = {}
        group_lime_values = {}
    
        for group in groups:
            group_mask = (sensitive == group)
            group_X = X[group_mask]
            sample_size = min(100, len(group_X))
            if sample_size < len(group_X):
                indices = np.random.choice(len(group_X), sample_size, replace=False)
                group_X = group_X[indices]
            # Compute SHAP values for the group.
            group_shap = shap_explainer(group_X)
            group_shap_values[group] = np.abs(group_shap.values).mean(axis=0)
        
            # Compute LIME explanations for the group.
            # For LIME, we compute explanation for each instance and average.
            lime_expls = []
            for instance in group_X:
                lime_exp = lime_explainer.explain_instance(
                    data_row=instance,
                    predict_fn=lambda x: model.predict(x),
                    num_features=X.shape[1],
                    top_labels=1
                )
                lime_map = lime_exp.as_map()
                # Use the first available key.
                key = list(lime_map.keys())[0] if lime_map else None
                # Create a vector with zeros and fill in with the weights.
                lime_vec = np.zeros(X.shape[1])
                if key is not None:
                    for feat_idx, weight in lime_map[key]:
                        lime_vec[feat_idx] = weight
                lime_expls.append(np.abs(lime_vec))
            group_lime_values[group] = np.mean(lime_expls, axis=0)
    
        # Plot grouped bar chart.
        x = np.arange(X.shape[1])
        width = 0.35  # width of the bars
    
        fig, ax = plt.subplots(figsize=(12, 8))
        for i, group in enumerate(groups):
            shap_vals = group_shap_values[group]
            lime_vals = group_lime_values[group]
            ax.bar(x + i*width - width/2, shap_vals, width/2, label=f'Group {group} SHAP')
            ax.bar(x + i*width + width/2, lime_vals, width/2, label=f'Group {group} LIME')
    
        ax.set_xlabel('Features')
        ax.set_ylabel('Average Absolute Attribution')
        ax.set_title('Feature Importance by Group (SHAP vs LIME)')
        ax.set_xticks(x + width * (len(groups)-1)/2)
        ax.set_xticklabels([f'F{i}' for i in range(X.shape[1])])
        ax.legend()
        plt.tight_layout()
        plt.savefig('feature_importance_by_group_shap_lime_bail.png')

    analyze_explanations_by_group(final_model, X_test, sens_test)

    def visualize_stability_distributions(model, X, sensitive):
        # Use the same parameters as calculate_gesd
        num_perturbations = 10
        perturbation_scale = 0.1
        mask_prob = 0.2
        baseline_value = 0.0

        # Compute per-instance stability scores (SHAP & LIME) exactly as in calculate_gesd
        n_features = X.shape[1]
        background = X[np.random.choice(X.shape[0], min(100, X.shape[0]), replace=False)]
        shap_explainer = shap.DeepExplainer(model, background)
        lime_explainer = LimeTabularExplainer(
            training_data=background,
            feature_names=None,
            class_names=['class0','class1'],
            mode='classification'
        )

        stabilities_shap = []
        stabilities_lime = []
        groups = []

        # Sample up to 200 instances for speed
        indices = np.random.choice(X.shape[0], min(200, X.shape[0]), replace=False)
        for idx in indices:
            instance = X[idx]
            group = sensitive[idx]

            # get baseline explanations
            shap_vals = shap_explainer.shap_values(instance.reshape(1, -1))
            shap_vec = np.squeeze(shap_vals[0] if isinstance(shap_vals, list) else shap_vals)
            if shap_vec.ndim != 1:
                shap_vec = shap_vec.flatten()

            def predict_fn(z):
                p = model.predict(z)[0]
                arr = np.zeros((z.shape[0], 2))
                arr[:,0] = 1 - p.flatten()
                arr[:,1] = p.flatten()
                return arr

            lime_exp = lime_explainer.explain_instance(instance, predict_fn, num_features=n_features, top_labels=1)
            lm = lime_exp.as_map()
            lime_map = lm[list(lm.keys())[0]] if lm else []
            lime_vec = np.zeros(n_features)
            for fi, w in lime_map:
                lime_vec[fi] = w

            # now perturb
            shap_scores, lime_scores = [], []
            for _ in range(num_perturbations):
                pert = instance + np.random.normal(0, perturbation_scale, size=instance.shape)
                mask = np.random.rand(n_features) < mask_prob
                pert[mask] = baseline_value

                # SHAP
                pv = shap_explainer.shap_values(pert.reshape(1, -1))
                pv0 = np.squeeze(pv[0] if isinstance(pv, list) else pv)
                if pv0.ndim != 1: pv0 = pv0.flatten()
                dist_s = np.linalg.norm(shap_vec - pv0, ord=1)
                shap_scores.append(1 / (1 + dist_s))

                # LIME
                lime_p = lime_explainer.explain_instance(pert, predict_fn, num_features=n_features, top_labels=1).as_map()
                lm_p = lime_p[list(lime_p.keys())[0]] if lime_p else []
                lv = np.zeros(n_features)
                for fi, w in lm_p:
                    lv[fi] = w
                dist_l = np.linalg.norm(lime_vec - lv, ord=1)
                lime_scores.append(1 / (1 + dist_l))

            stabilities_shap.append(np.mean(shap_scores))
            stabilities_lime.append(np.mean(lime_scores))
            groups.append(group)

        stabilities_shap = np.array(stabilities_shap)
        stabilities_lime = np.array(stabilities_lime)
        groups = np.array(groups)
        combined = (stabilities_shap + stabilities_lime) / 2.0

        # Plot
        unique = np.unique(groups)
        fig, axes = plt.subplots(1, 3, figsize=(18,6), sharey=True)
        for g in unique:
            mask = groups == g
            axes[0].hist(stabilities_shap[mask], bins=15, alpha=0.6, label=f'Group {g}')
            axes[1].hist(stabilities_lime[mask], bins=15, alpha=0.6, label=f'Group {g}')
            axes[2].hist(combined[mask], bins=15, alpha=0.6, label=f'Group {g}')

        axes[0].set_title("SHAP Stability")
        axes[1].set_title("LIME Stability")
        axes[2].set_title("Combined Stability")
        for ax, lbl in zip(axes, ['Score', 'Score', 'Score']):
            ax.set_xlabel(lbl)
            ax.legend()
        axes[0].set_ylabel("Count")

        plt.tight_layout()
        plt.savefig('stability_distribution_by_group_combined_bail.png')

    
        for group in unique:
            grp_shap = stabilities_shap[groups == group]
            grp_lime = stabilities_lime[groups == group]
            grp_comb = combined[groups == group]

            print(f"Group {group} SHAP stability:     mean={np.mean(grp_shap):.4f}, std={np.std(grp_shap):.4f}")
            print(f"Group {group} LIME stability:     mean={np.mean(grp_lime):.4f}, std={np.std(grp_lime):.4f}")
            print(f"Group {group} Combined stability: mean={np.mean(grp_comb):.4f}, std={np.std(grp_comb):.4f}")
    # Visualize stability distributions for the best model
    print("\nExplanation stability statistics by group:")
    visualize_stability_distributions(final_model, X_test, sens_test)
if __name__ == "__main__":
    main()
