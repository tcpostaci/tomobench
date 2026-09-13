"""Small interpretable baseline models for bounded ML experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

PLANNED_BASELINE_MODELS = (
    "mean_target_regressor",
    "feature_distance_weighted_target_regressor",
    "linear_regression",
    "random_forest",
    "pca_linear_observation_regressor",
    "pca_ridge_observation_regressor",
    "pca_mlp_observation_regressor",
    "pca_random_forest_observation_regressor",
)


def planned_target_representations() -> tuple[str, ...]:
    """Return candidate target representations for future model work."""
    return ("velocity_grid", "scenario_parameters")


@dataclass(frozen=True)
class MeanTargetRegressor:
    """Predict the element-wise mean target vector from the training set."""

    mean_target_vector: tuple[float, ...]

    @classmethod
    def fit(cls, target_vectors: Sequence[Sequence[float]]) -> "MeanTargetRegressor":
        if not target_vectors:
            raise ValueError("MeanTargetRegressor requires at least one training target vector.")
        vector_length = len(target_vectors[0])
        if vector_length == 0:
            raise ValueError("Target vectors must not be empty.")
        if any(len(vector) != vector_length for vector in target_vectors):
            raise ValueError("All target vectors must have the same length.")
        mean_vector = tuple(
            sum(vector[index] for vector in target_vectors) / len(target_vectors)
            for index in range(vector_length)
        )
        return cls(mean_target_vector=mean_vector)

    def predict(self) -> tuple[float, ...]:
        """Return the learned mean target vector."""
        return self.mean_target_vector


@dataclass(frozen=True)
class FeatureDistanceWeightedTargetRegressor:
    """Predict by inverse-distance weighting in standardized feature space."""

    training_feature_vectors: tuple[tuple[float, ...], ...]
    training_target_vectors: tuple[tuple[float, ...], ...]
    feature_means: tuple[float, ...]
    feature_stds: tuple[float, ...]

    @classmethod
    def fit(
        cls,
        feature_vectors: Sequence[Sequence[float]],
        target_vectors: Sequence[Sequence[float]],
    ) -> "FeatureDistanceWeightedTargetRegressor":
        if not feature_vectors or not target_vectors:
            raise ValueError("FeatureDistanceWeightedTargetRegressor requires training data.")
        if len(feature_vectors) != len(target_vectors):
            raise ValueError("Feature and target counts must match.")
        feature_length = len(feature_vectors[0])
        target_length = len(target_vectors[0])
        if any(len(vector) != feature_length for vector in feature_vectors):
            raise ValueError("All feature vectors must have the same length.")
        if any(len(vector) != target_length for vector in target_vectors):
            raise ValueError("All target vectors must have the same length.")
        feature_means = tuple(
            sum(float(vector[index]) for vector in feature_vectors) / len(feature_vectors)
            for index in range(feature_length)
        )
        feature_stds = tuple(
            _population_std([float(vector[index]) for vector in feature_vectors])
            for index in range(feature_length)
        )
        return cls(
            training_feature_vectors=tuple(
                tuple(float(value) for value in vector) for vector in feature_vectors
            ),
            training_target_vectors=tuple(
                tuple(float(value) for value in vector) for vector in target_vectors
            ),
            feature_means=feature_means,
            feature_stds=feature_stds,
        )

    def predict(
        self,
        feature_vector: Sequence[float],
    ) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
        """Return the weighted prediction, training distances, and training weights."""
        if len(feature_vector) != len(self.feature_means):
            raise ValueError("Feature vector length does not match fitted feature space.")
        standardized_query = _standardize(
            feature_vector,
            self.feature_means,
            self.feature_stds,
        )
        distances = tuple(
            _euclidean_distance(
                standardized_query,
                _standardize(training_vector, self.feature_means, self.feature_stds),
            )
            for training_vector in self.training_feature_vectors
        )
        zero_distance_index = next(
            (index for index, distance in enumerate(distances) if distance <= 1.0e-12),
            None,
        )
        if zero_distance_index is not None:
            weights = tuple(
                1.0 if index == zero_distance_index else 0.0
                for index in range(len(self.training_target_vectors))
            )
            return self.training_target_vectors[zero_distance_index], distances, weights
        inverse_distances = tuple(1.0 / distance for distance in distances)
        weight_sum = sum(inverse_distances)
        weights = tuple(weight / weight_sum for weight in inverse_distances)
        prediction = tuple(
            sum(
                weight * target_vector[index]
                for weight, target_vector in zip(
                    weights,
                    self.training_target_vectors,
                    strict=True,
                )
            )
            for index in range(len(self.training_target_vectors[0]))
        )
        return prediction, distances, weights


def _population_std(values: Sequence[float]) -> float:
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _standardize(
    values: Sequence[float],
    means: Sequence[float],
    stds: Sequence[float],
) -> tuple[float, ...]:
    standardized: list[float] = []
    for value, mean_value, std_value in zip(values, means, stds, strict=True):
        if std_value <= 1.0e-12:
            standardized.append(0.0)
        else:
            standardized.append((float(value) - mean_value) / std_value)
    return tuple(standardized)


def _euclidean_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(
        sum(
            (left_value - right_value) ** 2
            for left_value, right_value in zip(left, right, strict=True)
        )
    )


@dataclass(frozen=True)
class RidgeReducedTargetRegressor:
    """Multi-output ridge regressor for small reduced-target vectors."""

    coefficients_by_output: tuple[tuple[float, ...], ...]
    intercepts_by_output: tuple[float, ...]
    feature_means: tuple[float, ...]
    feature_stds: tuple[float, ...]

    @classmethod
    def fit(
        cls,
        feature_vectors: Sequence[Sequence[float]],
        target_vectors: Sequence[Sequence[float]],
        ridge_alpha: float,
    ) -> "RidgeReducedTargetRegressor":
        if not feature_vectors or not target_vectors:
            raise ValueError("RidgeReducedTargetRegressor requires training data.")
        if len(feature_vectors) != len(target_vectors):
            raise ValueError("Feature and target counts must match.")
        if ridge_alpha <= 0.0:
            raise ValueError("ridge_alpha must be positive.")
        feature_length = len(feature_vectors[0])
        target_length = len(target_vectors[0])
        if any(len(vector) != feature_length for vector in feature_vectors):
            raise ValueError("All feature vectors must have the same length.")
        if any(len(vector) != target_length for vector in target_vectors):
            raise ValueError("All target vectors must have the same length.")

        feature_means = tuple(
            sum(float(vector[index]) for vector in feature_vectors) / len(feature_vectors)
            for index in range(feature_length)
        )
        feature_stds = tuple(
            _population_std([float(vector[index]) for vector in feature_vectors])
            for index in range(feature_length)
        )
        standardized_features = [
            _standardize(vector, feature_means, feature_stds) for vector in feature_vectors
        ]
        design_matrix = [list(vector) + [1.0] for vector in standardized_features]
        normal_matrix = _normal_matrix(design_matrix, ridge_alpha)

        coefficients_by_output: list[tuple[float, ...]] = []
        intercepts_by_output: list[float] = []
        for output_index in range(target_length):
            target_column = [float(vector[output_index]) for vector in target_vectors]
            right_hand_side = _design_transpose_times_vector(design_matrix, target_column)
            solution = _solve_linear_system(normal_matrix, right_hand_side)
            coefficients_by_output.append(tuple(solution[:-1]))
            intercepts_by_output.append(solution[-1])
        return cls(
            coefficients_by_output=tuple(coefficients_by_output),
            intercepts_by_output=tuple(intercepts_by_output),
            feature_means=feature_means,
            feature_stds=feature_stds,
        )

    def predict(self, feature_vector: Sequence[float]) -> tuple[float, ...]:
        """Predict one reduced target vector."""
        if len(feature_vector) != len(self.feature_means):
            raise ValueError("Feature vector length does not match fitted feature space.")
        standardized = _standardize(feature_vector, self.feature_means, self.feature_stds)
        predictions: list[float] = []
        for coefficients, intercept in zip(
            self.coefficients_by_output,
            self.intercepts_by_output,
            strict=True,
        ):
            predictions.append(
                sum(
                    weight * value for weight, value in zip(coefficients, standardized, strict=True)
                )
                + intercept
            )
        return tuple(predictions)


@dataclass(frozen=True)
class PCARidgeObservationRegressor:
    """Predict full frozen target grids from observation-level case features."""

    training_feature_vectors: tuple[tuple[float, ...], ...]
    dual_coefficients_by_component: tuple[tuple[float, ...], ...]
    feature_means: tuple[float, ...]
    feature_stds: tuple[float, ...]
    target_mean_vector: tuple[float, ...]
    principal_components: tuple[tuple[float, ...], ...]
    velocity_bounds: tuple[float, float]

    @classmethod
    def fit(
        cls,
        feature_vectors: Sequence[Sequence[float]],
        target_vectors: Sequence[Sequence[float]],
        pca_component_count: int,
        ridge_alpha: float,
        velocity_bounds: tuple[float, float],
    ) -> "PCARidgeObservationRegressor":
        if not feature_vectors or not target_vectors:
            raise ValueError("PCARidgeObservationRegressor requires training data.")
        if len(feature_vectors) != len(target_vectors):
            raise ValueError("Feature and target counts must match.")
        if pca_component_count <= 0:
            raise ValueError("pca_component_count must be positive.")
        if ridge_alpha <= 0.0:
            raise ValueError("ridge_alpha must be positive.")
        feature_length = len(feature_vectors[0])
        target_length = len(target_vectors[0])
        if any(len(vector) != feature_length for vector in feature_vectors):
            raise ValueError("All feature vectors must have the same length.")
        if any(len(vector) != target_length for vector in target_vectors):
            raise ValueError("All target vectors must have the same length.")
        if len(target_vectors) < 2:
            raise ValueError("PCARidgeObservationRegressor requires at least two training cases.")

        feature_means = tuple(
            sum(float(vector[index]) for vector in feature_vectors) / len(feature_vectors)
            for index in range(feature_length)
        )
        feature_stds = tuple(
            _population_std([float(vector[index]) for vector in feature_vectors])
            for index in range(feature_length)
        )
        standardized_features = tuple(
            _standardize(vector, feature_means, feature_stds) for vector in feature_vectors
        )
        target_projection = _fit_target_pca_projection(
            target_vectors=target_vectors,
            component_count=pca_component_count,
        )
        kernel_matrix = _gram_matrix(standardized_features)
        ridge_matrix = [
            [
                kernel_matrix[row_index][column_index]
                + (ridge_alpha if row_index == column_index else 0.0)
                for column_index in range(len(kernel_matrix))
            ]
            for row_index in range(len(kernel_matrix))
        ]
        dual_coefficients_by_component: list[tuple[float, ...]] = []
        for component_index in range(len(target_projection.principal_components)):
            right_hand_side = [
                target_projection.coefficient_matrix[row_index][component_index]
                for row_index in range(len(target_projection.coefficient_matrix))
            ]
            dual_coefficients_by_component.append(
                tuple(_solve_linear_system(ridge_matrix, right_hand_side))
            )
        return cls(
            training_feature_vectors=standardized_features,
            dual_coefficients_by_component=tuple(dual_coefficients_by_component),
            feature_means=feature_means,
            feature_stds=feature_stds,
            target_mean_vector=target_projection.target_mean_vector,
            principal_components=tuple(target_projection.principal_components),
            velocity_bounds=velocity_bounds,
        )

    def predict(self, feature_vector: Sequence[float]) -> tuple[float, ...]:
        """Predict one full frozen target grid."""
        if len(feature_vector) != len(self.feature_means):
            raise ValueError("Feature vector length does not match fitted feature space.")
        standardized = _standardize(feature_vector, self.feature_means, self.feature_stds)
        kernel_vector = tuple(
            sum(left * right for left, right in zip(standardized, training_vector, strict=True))
            for training_vector in self.training_feature_vectors
        )
        predicted_coefficients = tuple(
            sum(
                weight * dual_value
                for weight, dual_value in zip(kernel_vector, dual_values, strict=True)
            )
            for dual_values in self.dual_coefficients_by_component
        )
        prediction: list[float] = []
        for target_index, mean_value in enumerate(self.target_mean_vector):
            reconstructed = mean_value + sum(
                coefficient * self.principal_components[component_index][target_index]
                for component_index, coefficient in enumerate(predicted_coefficients)
            )
            prediction.append(
                min(self.velocity_bounds[1], max(self.velocity_bounds[0], reconstructed))
            )
        return tuple(prediction)


@dataclass(frozen=True)
class PCALinearObservationRegressor:
    """Predict full frozen target grids with PCA plus minimum-norm linear regression."""

    training_feature_vectors: tuple[tuple[float, ...], ...]
    dual_coefficients_by_component: tuple[tuple[float, ...], ...]
    feature_means: tuple[float, ...]
    feature_stds: tuple[float, ...]
    target_mean_vector: tuple[float, ...]
    principal_components: tuple[tuple[float, ...], ...]
    velocity_bounds: tuple[float, float]

    @classmethod
    def fit(
        cls,
        feature_vectors: Sequence[Sequence[float]],
        target_vectors: Sequence[Sequence[float]],
        pca_component_count: int,
        velocity_bounds: tuple[float, float],
    ) -> "PCALinearObservationRegressor":
        if not feature_vectors or not target_vectors:
            raise ValueError("PCALinearObservationRegressor requires training data.")
        if len(feature_vectors) != len(target_vectors):
            raise ValueError("Feature and target counts must match.")
        if pca_component_count <= 0:
            raise ValueError("pca_component_count must be positive.")
        feature_length = len(feature_vectors[0])
        target_length = len(target_vectors[0])
        if any(len(vector) != feature_length for vector in feature_vectors):
            raise ValueError("All feature vectors must have the same length.")
        if any(len(vector) != target_length for vector in target_vectors):
            raise ValueError("All target vectors must have the same length.")
        if len(target_vectors) < 2:
            raise ValueError("PCALinearObservationRegressor requires at least two training cases.")

        feature_means = tuple(
            sum(float(vector[index]) for vector in feature_vectors) / len(feature_vectors)
            for index in range(feature_length)
        )
        feature_stds = tuple(
            _population_std([float(vector[index]) for vector in feature_vectors])
            for index in range(feature_length)
        )
        standardized_features = tuple(
            _standardize(vector, feature_means, feature_stds) for vector in feature_vectors
        )
        target_projection = _fit_target_pca_projection(
            target_vectors=target_vectors,
            component_count=pca_component_count,
        )
        kernel_matrix = _gram_matrix(standardized_features)
        dual_coefficients_by_component: list[tuple[float, ...]] = []
        for component_index in range(len(target_projection.principal_components)):
            right_hand_side = [
                target_projection.coefficient_matrix[row_index][component_index]
                for row_index in range(len(target_projection.coefficient_matrix))
            ]
            dual_coefficients_by_component.append(
                tuple(_solve_psd_pseudoinverse_system(kernel_matrix, right_hand_side))
            )
        return cls(
            training_feature_vectors=standardized_features,
            dual_coefficients_by_component=tuple(dual_coefficients_by_component),
            feature_means=feature_means,
            feature_stds=feature_stds,
            target_mean_vector=target_projection.target_mean_vector,
            principal_components=tuple(target_projection.principal_components),
            velocity_bounds=velocity_bounds,
        )

    def predict(self, feature_vector: Sequence[float]) -> tuple[float, ...]:
        """Predict one full frozen target grid."""
        if len(feature_vector) != len(self.feature_means):
            raise ValueError("Feature vector length does not match fitted feature space.")
        standardized = _standardize(feature_vector, self.feature_means, self.feature_stds)
        kernel_vector = tuple(
            sum(left * right for left, right in zip(standardized, training_vector, strict=True))
            for training_vector in self.training_feature_vectors
        )
        predicted_coefficients = tuple(
            sum(
                weight * dual_value
                for weight, dual_value in zip(kernel_vector, dual_values, strict=True)
            )
            for dual_values in self.dual_coefficients_by_component
        )
        return _reconstruct_target_from_components(
            predicted_coefficients,
            self.target_mean_vector,
            self.principal_components,
            self.velocity_bounds,
        )


@dataclass(frozen=True)
class PCAMLPObservationRegressor:
    """Predict full frozen target grids with a shallow MLP over PCA target coefficients."""

    feature_means: tuple[float, ...]
    feature_stds: tuple[float, ...]
    hidden_biases: tuple[float, ...]
    hidden_weights: tuple[tuple[float, ...], ...]
    output_biases: tuple[float, ...]
    output_weights: tuple[tuple[float, ...], ...]
    coefficient_means: tuple[float, ...]
    coefficient_stds: tuple[float, ...]
    target_mean_vector: tuple[float, ...]
    principal_components: tuple[tuple[float, ...], ...]
    velocity_bounds: tuple[float, float]

    @classmethod
    def fit(
        cls,
        feature_vectors: Sequence[Sequence[float]],
        target_vectors: Sequence[Sequence[float]],
        pca_component_count: int,
        hidden_width: int,
        epoch_count: int,
        learning_rate: float,
        l2_alpha: float,
        velocity_bounds: tuple[float, float],
        random_seed: int,
    ) -> "PCAMLPObservationRegressor":
        if not feature_vectors or not target_vectors:
            raise ValueError("PCAMLPObservationRegressor requires training data.")
        if len(feature_vectors) != len(target_vectors):
            raise ValueError("Feature and target counts must match.")
        if pca_component_count <= 0:
            raise ValueError("pca_component_count must be positive.")
        if hidden_width <= 0:
            raise ValueError("hidden_width must be positive.")
        if epoch_count <= 0:
            raise ValueError("epoch_count must be positive.")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if l2_alpha < 0.0:
            raise ValueError("l2_alpha must be non-negative.")
        feature_length = len(feature_vectors[0])
        target_length = len(target_vectors[0])
        if any(len(vector) != feature_length for vector in feature_vectors):
            raise ValueError("All feature vectors must have the same length.")
        if any(len(vector) != target_length for vector in target_vectors):
            raise ValueError("All target vectors must have the same length.")
        if len(target_vectors) < 2:
            raise ValueError("PCAMLPObservationRegressor requires at least two training cases.")

        feature_means = tuple(
            sum(float(vector[index]) for vector in feature_vectors) / len(feature_vectors)
            for index in range(feature_length)
        )
        feature_stds = tuple(
            _population_std([float(vector[index]) for vector in feature_vectors])
            for index in range(feature_length)
        )
        standardized_features = tuple(
            _standardize(vector, feature_means, feature_stds) for vector in feature_vectors
        )
        target_projection = _fit_target_pca_projection(
            target_vectors=target_vectors,
            component_count=pca_component_count,
        )
        coefficient_means = tuple(
            sum(row[index] for row in target_projection.coefficient_matrix)
            / len(target_projection.coefficient_matrix)
            for index in range(len(target_projection.principal_components))
        )
        coefficient_stds = tuple(
            _population_std([row[index] for row in target_projection.coefficient_matrix])
            for index in range(len(target_projection.principal_components))
        )
        standardized_coefficients = tuple(
            _standardize(row, coefficient_means, coefficient_stds)
            for row in target_projection.coefficient_matrix
        )

        rng = _baseline_rng(random_seed)
        hidden_weights = [
            [_uniform_weight(rng, feature_length) for _ in range(feature_length)]
            for _ in range(hidden_width)
        ]
        hidden_biases = [0.0 for _ in range(hidden_width)]
        output_weights = [
            [_uniform_weight(rng, hidden_width) for _ in range(hidden_width)]
            for _ in range(len(target_projection.principal_components))
        ]
        output_biases = [0.0 for _ in range(len(target_projection.principal_components))]

        sample_count = len(standardized_features)
        for _ in range(epoch_count):
            hidden_weight_grads = [
                [0.0 for _ in range(feature_length)] for _ in range(hidden_width)
            ]
            hidden_bias_grads = [0.0 for _ in range(hidden_width)]
            output_weight_grads = [
                [0.0 for _ in range(hidden_width)]
                for _ in range(len(target_projection.principal_components))
            ]
            output_bias_grads = [0.0 for _ in range(len(target_projection.principal_components))]
            for feature_vector, target_coefficients in zip(
                standardized_features,
                standardized_coefficients,
                strict=True,
            ):
                hidden_linear = [
                    sum(
                        weight * value
                        for weight, value in zip(weights, feature_vector, strict=True)
                    )
                    + bias
                    for weights, bias in zip(hidden_weights, hidden_biases, strict=True)
                ]
                hidden_activations = [math.tanh(value) for value in hidden_linear]
                output_values = [
                    sum(
                        weight * value
                        for weight, value in zip(weights, hidden_activations, strict=True)
                    )
                    + bias
                    for weights, bias in zip(output_weights, output_biases, strict=True)
                ]
                output_errors = [
                    output_value - target_value
                    for output_value, target_value in zip(
                        output_values,
                        target_coefficients,
                        strict=True,
                    )
                ]
                for output_index, error in enumerate(output_errors):
                    output_bias_grads[output_index] += error
                    for hidden_index, hidden_value in enumerate(hidden_activations):
                        output_weight_grads[output_index][hidden_index] += error * hidden_value
                hidden_errors = [0.0 for _ in range(hidden_width)]
                for hidden_index in range(hidden_width):
                    backprop_signal = sum(
                        output_errors[output_index] * output_weights[output_index][hidden_index]
                        for output_index in range(len(output_weights))
                    )
                    hidden_errors[hidden_index] = backprop_signal * (
                        1.0 - hidden_activations[hidden_index] * hidden_activations[hidden_index]
                    )
                for hidden_index, hidden_error in enumerate(hidden_errors):
                    hidden_bias_grads[hidden_index] += hidden_error
                    for feature_index, feature_value in enumerate(feature_vector):
                        hidden_weight_grads[hidden_index][feature_index] += (
                            hidden_error * feature_value
                        )
            learning_scale = learning_rate / sample_count
            for hidden_index in range(hidden_width):
                hidden_biases[hidden_index] -= learning_scale * hidden_bias_grads[hidden_index]
                for feature_index in range(feature_length):
                    hidden_weights[hidden_index][feature_index] -= learning_scale * (
                        hidden_weight_grads[hidden_index][feature_index]
                        + l2_alpha * hidden_weights[hidden_index][feature_index]
                    )
            for output_index in range(len(output_weights)):
                output_biases[output_index] -= learning_scale * output_bias_grads[output_index]
                for hidden_index in range(hidden_width):
                    output_weights[output_index][hidden_index] -= learning_scale * (
                        output_weight_grads[output_index][hidden_index]
                        + l2_alpha * output_weights[output_index][hidden_index]
                    )

        return cls(
            feature_means=feature_means,
            feature_stds=feature_stds,
            hidden_biases=tuple(hidden_biases),
            hidden_weights=tuple(tuple(row) for row in hidden_weights),
            output_biases=tuple(output_biases),
            output_weights=tuple(tuple(row) for row in output_weights),
            coefficient_means=coefficient_means,
            coefficient_stds=coefficient_stds,
            target_mean_vector=target_projection.target_mean_vector,
            principal_components=tuple(target_projection.principal_components),
            velocity_bounds=velocity_bounds,
        )

    def predict(self, feature_vector: Sequence[float]) -> tuple[float, ...]:
        """Predict one full frozen target grid."""
        if len(feature_vector) != len(self.feature_means):
            raise ValueError("Feature vector length does not match fitted feature space.")
        standardized = _standardize(feature_vector, self.feature_means, self.feature_stds)
        hidden_activations = [
            math.tanh(
                sum(weight * value for weight, value in zip(weights, standardized, strict=True))
                + bias
            )
            for weights, bias in zip(self.hidden_weights, self.hidden_biases, strict=True)
        ]
        standardized_coefficients = [
            sum(weight * value for weight, value in zip(weights, hidden_activations, strict=True))
            + bias
            for weights, bias in zip(self.output_weights, self.output_biases, strict=True)
        ]
        predicted_coefficients = [
            mean_value if std_value <= 1.0e-12 else mean_value + standardized_value * std_value
            for mean_value, std_value, standardized_value in zip(
                self.coefficient_means,
                self.coefficient_stds,
                standardized_coefficients,
                strict=True,
            )
        ]
        prediction: list[float] = []
        for target_index, mean_value in enumerate(self.target_mean_vector):
            reconstructed = mean_value + sum(
                coefficient * self.principal_components[component_index][target_index]
                for component_index, coefficient in enumerate(predicted_coefficients)
            )
            prediction.append(
                min(self.velocity_bounds[1], max(self.velocity_bounds[0], reconstructed))
            )
        return tuple(prediction)


@dataclass(frozen=True)
class PCARandomForestObservationRegressor:
    """Predict full frozen target grids with a small random forest over PCA coefficients."""

    feature_means: tuple[float, ...]
    feature_stds: tuple[float, ...]
    target_mean_vector: tuple[float, ...]
    principal_components: tuple[tuple[float, ...], ...]
    trees: tuple["_RandomForestNode", ...]
    velocity_bounds: tuple[float, float]

    @classmethod
    def fit(
        cls,
        feature_vectors: Sequence[Sequence[float]],
        target_vectors: Sequence[Sequence[float]],
        pca_component_count: int,
        tree_count: int,
        max_depth: int,
        min_samples_leaf: int,
        max_feature_count: int,
        velocity_bounds: tuple[float, float],
        random_seed: int,
    ) -> "PCARandomForestObservationRegressor":
        if not feature_vectors or not target_vectors:
            raise ValueError("PCARandomForestObservationRegressor requires training data.")
        if len(feature_vectors) != len(target_vectors):
            raise ValueError("Feature and target counts must match.")
        if pca_component_count <= 0:
            raise ValueError("pca_component_count must be positive.")
        if tree_count <= 0:
            raise ValueError("tree_count must be positive.")
        if max_depth <= 0:
            raise ValueError("max_depth must be positive.")
        if min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be positive.")
        if max_feature_count <= 0:
            raise ValueError("max_feature_count must be positive.")
        feature_length = len(feature_vectors[0])
        target_length = len(target_vectors[0])
        if any(len(vector) != feature_length for vector in feature_vectors):
            raise ValueError("All feature vectors must have the same length.")
        if any(len(vector) != target_length for vector in target_vectors):
            raise ValueError("All target vectors must have the same length.")
        if len(target_vectors) < 2:
            raise ValueError(
                "PCARandomForestObservationRegressor requires at least two training cases."
            )

        feature_means = tuple(
            sum(float(vector[index]) for vector in feature_vectors) / len(feature_vectors)
            for index in range(feature_length)
        )
        feature_stds = tuple(
            _population_std([float(vector[index]) for vector in feature_vectors])
            for index in range(feature_length)
        )
        standardized_features = tuple(
            _standardize(vector, feature_means, feature_stds) for vector in feature_vectors
        )
        target_projection = _fit_target_pca_projection(
            target_vectors=target_vectors,
            component_count=pca_component_count,
        )
        rng = _baseline_rng(random_seed)
        trees: list[_RandomForestNode] = []
        sample_count = len(standardized_features)
        capped_feature_count = min(max_feature_count, feature_length)
        for _ in range(tree_count):
            bootstrap_indices = [rng.randrange(sample_count) for _ in range(sample_count)]
            trees.append(
                _build_random_forest_tree(
                    feature_vectors=standardized_features,
                    target_vectors=target_projection.coefficient_matrix,
                    sample_indices=bootstrap_indices,
                    depth=0,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_feature_count=capped_feature_count,
                    rng=rng,
                )
            )
        return cls(
            feature_means=feature_means,
            feature_stds=feature_stds,
            target_mean_vector=target_projection.target_mean_vector,
            principal_components=tuple(target_projection.principal_components),
            trees=tuple(trees),
            velocity_bounds=velocity_bounds,
        )

    def predict(self, feature_vector: Sequence[float]) -> tuple[float, ...]:
        """Predict one full frozen target grid."""
        if len(feature_vector) != len(self.feature_means):
            raise ValueError("Feature vector length does not match fitted feature space.")
        standardized = _standardize(feature_vector, self.feature_means, self.feature_stds)
        predicted_coefficients = tuple(
            sum(tree.predict(standardized)[index] for tree in self.trees) / len(self.trees)
            for index in range(len(self.principal_components))
        )
        return _reconstruct_target_from_components(
            predicted_coefficients,
            self.target_mean_vector,
            self.principal_components,
            self.velocity_bounds,
        )


@dataclass(frozen=True)
class _TargetPCAProjection:
    target_mean_vector: tuple[float, ...]
    principal_components: tuple[tuple[float, ...], ...]
    coefficient_matrix: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class _RandomForestNode:
    prediction: tuple[float, ...]
    feature_index: int | None = None
    threshold: float | None = None
    left: "_RandomForestNode | None" = None
    right: "_RandomForestNode | None" = None

    def predict(self, feature_vector: Sequence[float]) -> tuple[float, ...]:
        if (
            self.feature_index is None
            or self.threshold is None
            or self.left is None
            or self.right is None
        ):
            return self.prediction
        if feature_vector[self.feature_index] <= self.threshold:
            return self.left.predict(feature_vector)
        return self.right.predict(feature_vector)


def _normal_matrix(
    design_matrix: Sequence[Sequence[float]], ridge_alpha: float
) -> list[list[float]]:
    column_count = len(design_matrix[0])
    matrix = [[0.0 for _ in range(column_count)] for _ in range(column_count)]
    for row in design_matrix:
        for left_index in range(column_count):
            for right_index in range(column_count):
                matrix[left_index][right_index] += row[left_index] * row[right_index]
    for index in range(column_count - 1):
        matrix[index][index] += ridge_alpha
    return matrix


def _design_transpose_times_vector(
    design_matrix: Sequence[Sequence[float]],
    vector: Sequence[float],
) -> list[float]:
    column_count = len(design_matrix[0])
    result = [0.0 for _ in range(column_count)]
    for row, value in zip(design_matrix, vector, strict=True):
        for index in range(column_count):
            result[index] += row[index] * value
    return result


def _solve_linear_system(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    size = len(vector)
    augmented = [list(matrix[row_index]) + [float(vector[row_index])] for row_index in range(size)]
    for pivot_index in range(size):
        max_row_index = max(
            range(pivot_index, size),
            key=lambda row_index: abs(augmented[row_index][pivot_index]),
        )
        if abs(augmented[max_row_index][pivot_index]) <= 1.0e-12:
            raise ValueError("Linear system is singular.")
        augmented[pivot_index], augmented[max_row_index] = (
            augmented[max_row_index],
            augmented[pivot_index],
        )
        pivot_value = augmented[pivot_index][pivot_index]
        augmented[pivot_index] = [value / pivot_value for value in augmented[pivot_index]]
        for row_index in range(size):
            if row_index == pivot_index:
                continue
            factor = augmented[row_index][pivot_index]
            augmented[row_index] = [
                row_value - factor * pivot_row_value
                for row_value, pivot_row_value in zip(
                    augmented[row_index],
                    augmented[pivot_index],
                    strict=True,
                )
            ]
    return [augmented[row_index][-1] for row_index in range(size)]


def _solve_psd_pseudoinverse_system(
    matrix: Sequence[Sequence[float]],
    vector: Sequence[float],
) -> list[float]:
    eigenvalues, eigenvectors = _jacobi_eigendecomposition(matrix)
    solution = [0.0 for _ in range(len(vector))]
    for eigen_index, eigenvalue in enumerate(eigenvalues):
        if eigenvalue <= 1.0e-10:
            continue
        basis_vector = [eigenvectors[row_index][eigen_index] for row_index in range(len(vector))]
        projection = sum(
            basis_value * float(target_value)
            for basis_value, target_value in zip(basis_vector, vector, strict=True)
        )
        scaled_projection = projection / eigenvalue
        for row_index, basis_value in enumerate(basis_vector):
            solution[row_index] += basis_value * scaled_projection
    return solution


def _gram_matrix(vectors: Sequence[Sequence[float]]) -> list[list[float]]:
    matrix = [[0.0 for _ in range(len(vectors))] for _ in range(len(vectors))]
    for left_index, left_vector in enumerate(vectors):
        for right_index in range(left_index, len(vectors)):
            dot_product = sum(
                left_value * right_value
                for left_value, right_value in zip(left_vector, vectors[right_index], strict=True)
            )
            matrix[left_index][right_index] = dot_product
            matrix[right_index][left_index] = dot_product
    return matrix


def _fit_target_pca_projection(
    target_vectors: Sequence[Sequence[float]],
    component_count: int,
) -> _TargetPCAProjection:
    target_length = len(target_vectors[0])
    target_mean_vector = tuple(
        sum(float(vector[index]) for vector in target_vectors) / len(target_vectors)
        for index in range(target_length)
    )
    centered_targets = tuple(
        tuple(
            float(value) - mean_value
            for value, mean_value in zip(vector, target_mean_vector, strict=True)
        )
        for vector in target_vectors
    )
    principal_components = _principal_components_from_targets(
        centered_targets=centered_targets,
        component_count=component_count,
    )
    if not principal_components:
        raise ValueError("PCA could not extract any non-zero target components.")
    coefficient_matrix = tuple(
        tuple(
            sum(
                value * component_value
                for value, component_value in zip(target, component, strict=True)
            )
            for component in principal_components
        )
        for target in centered_targets
    )
    return _TargetPCAProjection(
        target_mean_vector=target_mean_vector,
        principal_components=tuple(principal_components),
        coefficient_matrix=coefficient_matrix,
    )


def _baseline_rng(seed: int):
    import random

    return random.Random(seed)


def _uniform_weight(rng, fan_in: int) -> float:
    scale = 1.0 / math.sqrt(max(1, fan_in))
    return rng.uniform(-scale, scale)


def _reconstruct_target_from_components(
    coefficients: Sequence[float],
    target_mean_vector: Sequence[float],
    principal_components: Sequence[Sequence[float]],
    velocity_bounds: tuple[float, float],
) -> tuple[float, ...]:
    prediction: list[float] = []
    for target_index, mean_value in enumerate(target_mean_vector):
        reconstructed = mean_value + sum(
            coefficient * principal_components[component_index][target_index]
            for component_index, coefficient in enumerate(coefficients)
        )
        prediction.append(min(velocity_bounds[1], max(velocity_bounds[0], reconstructed)))
    return tuple(prediction)


def _build_random_forest_tree(
    *,
    feature_vectors: Sequence[Sequence[float]],
    target_vectors: Sequence[Sequence[float]],
    sample_indices: Sequence[int],
    depth: int,
    max_depth: int,
    min_samples_leaf: int,
    max_feature_count: int,
    rng,
) -> _RandomForestNode:
    prediction = _mean_vector(target_vectors, sample_indices)
    if (
        depth >= max_depth
        or len(sample_indices) <= min_samples_leaf * 2
        or _target_variance(target_vectors, sample_indices) <= 1.0e-12
    ):
        return _RandomForestNode(prediction=prediction)

    feature_length = len(feature_vectors[0])
    feature_indices = list(range(feature_length))
    rng.shuffle(feature_indices)
    candidate_features = feature_indices[: min(max_feature_count, feature_length)]
    best_split: tuple[float, int, float, list[int], list[int]] | None = None
    for feature_index in candidate_features:
        thresholds = _candidate_thresholds(feature_vectors, sample_indices, feature_index)
        for threshold in thresholds:
            left_indices = [
                index
                for index in sample_indices
                if feature_vectors[index][feature_index] <= threshold
            ]
            right_indices = [
                index
                for index in sample_indices
                if feature_vectors[index][feature_index] > threshold
            ]
            if len(left_indices) < min_samples_leaf or len(right_indices) < min_samples_leaf:
                continue
            split_score = _target_sse(target_vectors, left_indices) + _target_sse(
                target_vectors,
                right_indices,
            )
            if best_split is None or split_score < best_split[0]:
                best_split = (
                    split_score,
                    feature_index,
                    threshold,
                    left_indices,
                    right_indices,
                )
    if best_split is None:
        return _RandomForestNode(prediction=prediction)
    _, feature_index, threshold, left_indices, right_indices = best_split
    return _RandomForestNode(
        prediction=prediction,
        feature_index=feature_index,
        threshold=threshold,
        left=_build_random_forest_tree(
            feature_vectors=feature_vectors,
            target_vectors=target_vectors,
            sample_indices=left_indices,
            depth=depth + 1,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_feature_count=max_feature_count,
            rng=rng,
        ),
        right=_build_random_forest_tree(
            feature_vectors=feature_vectors,
            target_vectors=target_vectors,
            sample_indices=right_indices,
            depth=depth + 1,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_feature_count=max_feature_count,
            rng=rng,
        ),
    )


def _candidate_thresholds(
    feature_vectors: Sequence[Sequence[float]],
    sample_indices: Sequence[int],
    feature_index: int,
) -> tuple[float, ...]:
    sorted_values = sorted({feature_vectors[index][feature_index] for index in sample_indices})
    if len(sorted_values) <= 1:
        return ()
    return tuple(
        (sorted_values[index] + sorted_values[index + 1]) / 2.0
        for index in range(len(sorted_values) - 1)
    )


def _mean_vector(
    target_vectors: Sequence[Sequence[float]],
    sample_indices: Sequence[int],
) -> tuple[float, ...]:
    vector_length = len(target_vectors[0])
    return tuple(
        sum(target_vectors[index][component_index] for index in sample_indices)
        / len(sample_indices)
        for component_index in range(vector_length)
    )


def _target_sse(
    target_vectors: Sequence[Sequence[float]],
    sample_indices: Sequence[int],
) -> float:
    mean_vector = _mean_vector(target_vectors, sample_indices)
    return sum(
        sum(
            (target_vectors[index][component_index] - mean_vector[component_index]) ** 2
            for component_index in range(len(mean_vector))
        )
        for index in sample_indices
    )


def _target_variance(
    target_vectors: Sequence[Sequence[float]],
    sample_indices: Sequence[int],
) -> float:
    return _target_sse(target_vectors, sample_indices) / len(sample_indices)


def _principal_components_from_targets(
    centered_targets: Sequence[Sequence[float]],
    component_count: int,
) -> tuple[tuple[float, ...], ...]:
    gram_matrix = _gram_matrix(centered_targets)
    eigenvalues, eigenvectors = _jacobi_eigendecomposition(gram_matrix)
    ranked = sorted(
        (
            (
                eigenvalues[index],
                tuple(eigenvectors[row_index][index] for row_index in range(len(eigenvectors))),
            )
            for index in range(len(eigenvalues))
            if eigenvalues[index] > 1.0e-10
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    components: list[tuple[float, ...]] = []
    for eigenvalue, sample_space_vector in ranked[:component_count]:
        component = []
        for target_index in range(len(centered_targets[0])):
            projection = sum(
                sample_space_vector[sample_index] * centered_targets[sample_index][target_index]
                for sample_index in range(len(centered_targets))
            )
            component.append(projection / math.sqrt(eigenvalue))
        norm = math.sqrt(sum(value * value for value in component))
        if norm <= 1.0e-12:
            continue
        components.append(tuple(value / norm for value in component))
    return tuple(components)


def _jacobi_eigendecomposition(
    matrix: Sequence[Sequence[float]],
) -> tuple[list[float], list[list[float]]]:
    size = len(matrix)
    working = [list(row) for row in matrix]
    eigenvectors = [
        [1.0 if row_index == column_index else 0.0 for column_index in range(size)]
        for row_index in range(size)
    ]
    for _ in range(max(1, size * size * 20)):
        pivot_value = 0.0
        pivot_row = 0
        pivot_column = 1 if size > 1 else 0
        for row_index in range(size):
            for column_index in range(row_index + 1, size):
                candidate = abs(working[row_index][column_index])
                if candidate > pivot_value:
                    pivot_value = candidate
                    pivot_row = row_index
                    pivot_column = column_index
        if pivot_value <= 1.0e-12:
            break
        app = working[pivot_row][pivot_row]
        aqq = working[pivot_column][pivot_column]
        apq = working[pivot_row][pivot_column]
        tau = (aqq - app) / (2.0 * apq)
        tangent = math.copysign(1.0 / (abs(tau) + math.sqrt(1.0 + tau * tau)), tau)
        cosine = 1.0 / math.sqrt(1.0 + tangent * tangent)
        sine = tangent * cosine
        for index in range(size):
            if index in (pivot_row, pivot_column):
                continue
            aip = working[index][pivot_row]
            aiq = working[index][pivot_column]
            working[index][pivot_row] = cosine * aip - sine * aiq
            working[pivot_row][index] = working[index][pivot_row]
            working[index][pivot_column] = sine * aip + cosine * aiq
            working[pivot_column][index] = working[index][pivot_column]
        working[pivot_row][pivot_row] = (
            cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq
        )
        working[pivot_column][pivot_column] = (
            sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq
        )
        working[pivot_row][pivot_column] = 0.0
        working[pivot_column][pivot_row] = 0.0
        for index in range(size):
            vip = eigenvectors[index][pivot_row]
            viq = eigenvectors[index][pivot_column]
            eigenvectors[index][pivot_row] = cosine * vip - sine * viq
            eigenvectors[index][pivot_column] = sine * vip + cosine * viq
    eigenvalues = [working[index][index] for index in range(size)]
    return eigenvalues, eigenvectors
