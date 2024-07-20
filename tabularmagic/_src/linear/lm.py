import statsmodels.api as sm
from typing import Literal
from ..metrics.regression_scoring import RegressionScorer
from ..data.datahandler import DataEmitter


class OLSLinearModel:
    """Statsmodels OLS wrapper."""

    def __init__(self, name: str | None = None):
        """
        Initializes a OrdinaryLeastSquares object. Regresses y on X.

        Parameters
        ----------
        name : str.
            Default: None. Determines how the model shows up in the reports.
            If None, the name is set to be the class name.
        """
        self.estimator = None
        self._name = name
        if self._name is None:
            self._name = "OLS"

    def specify_data(self, dataemitter: DataEmitter):
        """Adds a DataEmitter object to the model.

        Parameters
        ----------
        dataemitter : DataEmitter containing all data. X and y variables
            must be specified.
        """
        self._dataemitter = dataemitter

    def fit(self):
        """Fits the model based on the data specified."""
        y_scaler = self._dataemitter.y_scaler()

        X_train, y_train = self._dataemitter.emit_train_Xy()
        n_predictors = X_train.shape[1]
        X_train = sm.add_constant(X_train)
        self.estimator = sm.OLS(y_train, X_train).fit(cov_type="HC3")

        y_pred_train = self.estimator.predict(X_train).to_numpy()
        if y_scaler is not None:
            y_pred_train = y_scaler.inverse_transform(y_pred_train)
            y_train = y_scaler.inverse_transform(y_train)

        self.train_scorer = RegressionScorer(
            y_pred=y_pred_train,
            y_true=y_train.to_numpy(),
            n_predictors=n_predictors,
            name=self._name,
        )

        X_test, y_test = self._dataemitter.emit_test_Xy()
        X_test = sm.add_constant(X_test)

        y_pred_test = self.estimator.predict(X_test).to_numpy()
        if y_scaler is not None:
            y_pred_test = y_scaler.inverse_transform(y_pred_test)
            y_test = y_scaler.inverse_transform(y_test)

        self.test_scorer = RegressionScorer(
            y_pred=y_pred_test,
            y_true=y_test.to_numpy(),
            n_predictors=n_predictors,
            name=self._name,
        )

    def _step(self, start_list = None,
              lower = [],
              upper = None,
              direction: Literal["both", "backward", "forward"] = "backward", 
              criteria: Literal["aic"] = "aic", 
              max_steps = 1000, verbose = False) -> list[str]:
        """Finish writing description

        Categorical variables will either be included or excluded as a whole.

        Parameters
        ----------
        criteria : str. Default: 'aic'.

        Returns
        -------
        list of str.
            The subset of predictors that are most likely to be significant.
        """
        if max_steps < 0:
            raise Exception("max_steps cannot be negative")
        
        X_train, y_train = self._dataemitter.emit_train_Xy()
        
        # Set upper to all possible variables if nothing is specified
        if upper is None:
            upper = X_train.columns.tolist()

        # If a starting list is not specified then choose defaults depending on
        # direction
        if start_list is None:
            if direction == "backward":
                start_list = upper.copy()
            else:
                start_list = lower.copy()
        
        # Check to see that the starting list and bounds agree
        if set(lower) >= set(start_list):
            raise Exception("starting list must include variables in lower bound")
        if set(upper) <= set(start_list):
            raise Exception("starting list contains variables not in upper bound")
        if set(lower) >= set(upper):
            raise Exception("lower bound is larger than upper bound")
        
        # Define a function to fit our OLS model given a list of features
        def fit_ols(feature_list):
            subset_X_train = X_train[feature_list]
            return sm.OLS(y_train, subset_X_train).fit(cov_type="HC3")
    
        # Create a function that fits and calculates the aic
        def calculate_aic(features):
            new_model = fit_ols(features)
            return new_model.aic, new_model

        # Set our current variables to our starting list
        included = start_list.copy()

        #Set our starting aic and best models
        current_aic, best_model = calculate_aic(included)

        current_step = 0
        while current_step < max_steps:
            # Keep track of whether or not something changed since last loop
            changed = False

            # Forward step
            if direction in ['forward', 'both']:
                excluded = list(set(upper) - set(included))
                aic_with_candidates = []
                for new_var in excluded:
                    candidate_features = included + [new_var]
                    aic, _ = calculate_aic(candidate_features)
                    aic_with_candidates.append((aic, new_var))
                aic_with_candidates.sort()
                best_add_aic, best_add_candidate = aic_with_candidates[0]


            # Backward step
            if direction in ['backward', 'both']:
                if len(included) > len(lower):
                    aic_with_candidates = []
                    for candidate in included:
                        if candidate not in lower:
                            candidate_features = included.copy()
                            candidate_features.remove(candidate)
                            aic, _ = calculate_aic(candidate_features)
                            aic_with_candidates.append((aic, candidate))
                    aic_with_candidates.sort()
                    # the best aic means the removed variable does not provide
                    # that much information
                    best_rem_aic, worst_rem_candidate = aic_with_candidates[0]


            # compare aic and update model if applicable
            if direction == 'forward' and best_add_aic < current_aic:
                included.append(best_add_candidate)
                current_aic = best_add_aic
                best_model = fit_ols(included)
                changed = True
                if verbose:
                    print(f"Adding: {best_add_candidate} with AIC {best_add_aic:.6}")
            elif direction == 'backward' and best_rem_aic < current_aic:
                included.remove(worst_rem_candidate)
                current_aic = best_rem_aic
                best_model = fit_ols(included)
                changed = True
                if verbose:
                    print(f"Drop {worst_rem_candidate} with AIC {best_rem_aic:.6}")
            elif direction == 'both':
                # First see if we want to remove or add a variable this step:
                if best_add_aic < best_rem_aic: # adding is better than removing
                    if best_add_aic < current_aic:
                        included.append(best_add_candidate)
                        current_aic = best_add_aic
                        best_model = fit_ols(included)
                        changed = True
                        if verbose:
                            print(f"Adding: {best_add_candidate} with AIC {best_add_aic:.6}")
                else: # removing is better than adding
                    if best_rem_aic < current_aic:
                        included.remove(worst_rem_candidate)
                        current_aic = best_rem_aic
                        best_model = fit_ols(included)
                        changed = True
                        if verbose:
                            print(f"Drop {worst_rem_candidate} with AIC {best_rem_aic:.6}")

            if not changed:
                break
            current_step += 1

        # Set the new model
        self.estimator = best_model

        return "" #placeholder change later



    def __str__(self):
        return self._name
