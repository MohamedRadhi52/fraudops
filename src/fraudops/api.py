"""Scoring API: fraud probability, decision and the three main reasons of each score.

The features come from the feature store upstream: the API receives them already computed.
"""

from pathlib import Path

import lightgbm
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/lightgbm.txt")
CONTROL_COST = 10  # euros per investigated card, see docs/cadrage.md
LABELS = {
    "tx_amount": "montant de la transaction",
    "customer_nb_tx_1d": "nombre de transactions du client sur 1 jour",
    "customer_avg_amount_1d": "montant moyen du client sur 1 jour",
    "customer_nb_tx_7d": "nombre de transactions du client sur 7 jours",
    "customer_avg_amount_7d": "montant moyen du client sur 7 jours",
    "customer_nb_tx_30d": "nombre de transactions du client sur 30 jours",
    "customer_avg_amount_30d": "montant moyen du client sur 30 jours",
    "terminal_nb_tx_1d": "transactions connues du terminal sur 1 jour",
    "terminal_risk_1d": "taux de fraude connu du terminal sur 1 jour",
    "terminal_nb_tx_7d": "transactions connues du terminal sur 7 jours",
    "terminal_risk_7d": "taux de fraude connu du terminal sur 7 jours",
    "terminal_nb_tx_30d": "transactions connues du terminal sur 30 jours",
    "terminal_risk_30d": "taux de fraude connu du terminal sur 30 jours",
}

model = lightgbm.Booster(model_file=MODEL_PATH)
app = FastAPI(title="FraudOps", description="Score de fraude des transactions carte")


class Transaction(BaseModel):
    tx_amount: float = Field(ge=0)
    customer_nb_tx_1d: int = Field(ge=1)
    customer_avg_amount_1d: float = Field(ge=0)
    customer_nb_tx_7d: int = Field(ge=1)
    customer_avg_amount_7d: float = Field(ge=0)
    customer_nb_tx_30d: int = Field(ge=1)
    customer_avg_amount_30d: float = Field(ge=0)
    terminal_nb_tx_1d: int = Field(ge=0)
    terminal_risk_1d: float = Field(ge=0, le=1)
    terminal_nb_tx_7d: int = Field(ge=0)
    terminal_risk_7d: float = Field(ge=0, le=1)
    terminal_nb_tx_30d: int = Field(ge=0)
    terminal_risk_30d: float = Field(ge=0, le=1)


class Reason(BaseModel):
    feature: str
    label: str
    contribution: float


class Score(BaseModel):
    probability: float
    expected_loss: float
    decision: str
    reasons: list[Reason]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/score")
def score(transaction: Transaction) -> Score:
    """Investigate the card when probability x amount exceeds the cost of a control."""
    features = model.feature_name()
    row = np.array([[getattr(transaction, name) for name in features]])
    probability = float(model.predict(row)[0])
    # SHAP contributions in log-odds, the last column is the bias
    contributions = model.predict(row, pred_contrib=True)[0, :-1]
    top = np.argsort(contributions)[::-1][:3]
    expected_loss = probability * transaction.tx_amount
    return Score(
        probability=probability,
        expected_loss=expected_loss,
        decision="contrôler" if expected_loss >= CONTROL_COST else "laisser passer",
        reasons=[
            Reason(feature=features[i], label=LABELS[features[i]], contribution=contributions[i])
            for i in top
        ],
    )
