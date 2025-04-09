import numpy as np

def split_dataset_test(target):
    X = np.array([[1, 10],
                  [2, 20],
                  [3, 30],
                  [4, 40]])
    
    feature_0 = 0
    node_indices = list(range(4))
    threshold_0 = 2

    feature_1 = 1
    threshold_1 = 35
    left_0, right_0 = target(X, node_indices, feature_0, threshold_0)
    left_1, right_1 = target(X, node_indices, feature_1, threshold_1)

    expected_0 = {'left': np.array([0, 1]),
                'right': np.array([2, 3])}
    
    expected_1 = {'left': np.array([0, 1, 2]),
                'right': np.array([3])}

    assert len(left_0) == 2, f"left must have 2 elements but got: {len(left_0)}"
    assert len(right_0) == 2, f"right must have 2 elements but got: {len(right_0)}"

    assert np.allclose(right_0, expected_0['right']), f"Wrong value for right. Expected: { expected_0['right']} \ngot: {right_0}"
    assert np.allclose(left_0, expected_0['left']), f"Wrong value for left. Expected: { expected_0['left']} \ngot: {left_0}"

    assert len(left_1) == 3, f"left must have 3 elements but got: {len(left_1)}"
    assert len(right_1) == 1, f"right must have 1 elements but got: {len(right_1)}"

    assert np.allclose(right_1, expected_1['right']), f"Wrong value for right. Expected: { expected_1['right']} \ngot: {right_1}"
    assert np.allclose(left_1, expected_1['left']), f"Wrong value for left. Expected: { expected_1['left']} \ngot: {left_1}"
    print("\033[92m 🌳 All tests passed for split_dataset.")
    
def compute_variance_reduction_test(target):
    
    X = np.array([[1, 10],
                  [2, 20],
                  [3, 30],
                  [4, 40]])
    y = np.array([15, 25, 35, 45])
            
    feature_0 = 0
    node_indices = list(range(4))
    threshold_0 = 2
    
    feature_1 = 1
    threshold_1 = 35

    # variance reduction 
    gain_0 = target(X, y, node_indices, feature_0, threshold_0)
    gain_1 = target(X, y, node_indices, feature_1, threshold_1)
    
    assert gain_0 > 0, f"Expected positive gain for feature 0, got: {gain_0}"
    assert gain_1 > 0, f"Expected positive gain for feature 1, got: {gain_1}"

    assert np.isclose(gain_0, 100.0, atol=1e-6), f"Wrong information gain. Expected {100.0} got: {gain_0}"
    assert np.isclose(gain_1, 75.0, atol=1e-6), f"Wrong information gain. Expected {75.0} got: {gain_1}"

    print("\033[92m 🌳 All tests passed for compute_variance_reduction.")

def get_thresholds_test(target):

    X=np.array([[15,800,9006],
                [27,25,1052],
                [36,589,4365],
                [38,269,856],
                [45,812,1234]])
    
    y=np.array([450,869,125,586,412])
    node_indices = list(range(5))
    
    feature_0 = 0
    thresholds_0 = target(X, node_indices, feature_0, max_thresholds = 20)
    
    feature_1 = 1
    thresholds_1 = target(X, node_indices, feature_1, max_thresholds = 20)
    
    feature_2 = 2
    thresholds_2 = target(X, node_indices, feature_2, max_thresholds = 20)

    assert np.allclose(thresholds_0, [21.0, 31.5, 37.0, 41.5]), \
        f"Wrong thresholds for feature 0. Expected [21.0, 31.5, 37.0, 41.5], got: {thresholds_0}"

    assert np.allclose(thresholds_1, [147.0, 429.0, 694.5, 806.0]), \
        f"Wrong thresholds for feature 1. Expected [147.0, 429.0, 694.5, 806.0], got: {thresholds_1}"

    assert np.allclose(thresholds_2, [954.0, 1143.0, 2799.5, 6685.5]), \
        f"Wrong thresholds for feature 2. Expected [954.0, 1143.0, 2799.5, 6685.5], got: {thresholds_2}"

    print("\033[92m 🌳 All tests passed for get_thresholds.")
    

























