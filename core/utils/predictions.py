import joblib

from sklearn.pipeline import Pipeline
from collections import Counter


def UnanimityPrediction(predictions_rf, predictions_gb, predictions_lr, predictions_mlp, predictions_xgb):
    final_predictions = []
    for rf, gb, lr, mlp, xgb in zip (predictions_rf, predictions_gb, predictions_lr, predictions_mlp, predictions_xgb):
        if rf == gb == lr == mlp == xgb:
            final_predictions.append(rf)
        else:
            final_predictions.append(1)
    return final_predictions


def MajorityVote(predictions_mlp, predictions_gbc, predictions_rfc):
    final_predictions = []
    for gbc, rfc, mlp in zip (predictions_gbc, predictions_rfc, predictions_mlp):
        count = Counter([gbc, rfc, mlp])
        top_pred = max(count, key=count.get)
        final_predictions.append(top_pred)
    return final_predictions