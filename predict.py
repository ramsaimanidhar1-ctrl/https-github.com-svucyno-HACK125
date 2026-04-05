import uuid
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("medalert.predict")

router = APIRouter()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    region: str = Field(..., min_length=2, max_length=100, examples=["AP-Nellore"])
    population: int = Field(..., gt=0, examples=[1500000])
    temperature_c: float = Field(..., ge=-50, le=60, examples=[34.2])
    humidity_pct: float = Field(..., ge=0, le=100, examples=[78.0])
    reported_cases_7d: int = Field(..., ge=0, examples=[42])

    @field_validator("region")
    @classmethod
    def sanitize_region(cls, v: str) -> str:
        return v.strip().title()


class FactorContribution(BaseModel):
    feature: str
    contribution: float


class PredictionResponse(BaseModel):
    prediction_id: str
    region: str
    risk_score: float = Field(..., ge=0, le=1)
    risk_level: str
    confidence: float = Field(..., ge=0, le=1)
    top_factors: list[FactorContribution]
    predicted_at: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _risk_level(score: float) -> str:
    if score >= 0.75:
        return "HIGH"
    if score >= 0.45:
        return "MEDIUM"
    return "LOW"


def _build_features(req: PredictionRequest) -> list[float]:
    """Convert request fields into the feature vector the model expects.
    Order must match the column order used during training."""
    return [
        req.population,
        req.temperature_c,
        req.humidity_pct,
        req.reported_cases_7d,
    ]


def _extract_top_factors(
    model, feature_vector: list[float]
) -> list[FactorContribution]:
    """Return top 3 SHAP-style feature contributions if the model exposes them,
    otherwise fall back to feature importances from tree-based models."""
    feature_names = ["population", "temperature_c", "humidity_pct", "reported_cases_7d"]

    try:
        # XGBoost / sklearn trees expose feature_importances_
        importances = model.feature_importances_
        paired = sorted(
            zip(feature_names, importances), key=lambda x: x[1], reverse=True
        )
        return [
            FactorContribution(feature=name, contribution=round(float(imp), 4))
            for name, imp in paired[:3]
        ]
    except AttributeError:
        # Model doesn't expose importances — return empty list gracefully
        logger.warning("Model does not expose feature_importances_; skipping factors.")
        return []


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post(
    "/",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict disease outbreak risk for a region",
)
async def predict(request: Request, payload: PredictionRequest):
    """
    Submit epidemiological data for a region and receive an AI-generated
    risk score (0 = no risk, 1 = highest risk) with contributing factors.
    """
    model = getattr(request.app.state, "model", None)
    if model is None:
        logger.error("Prediction requested but model is not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction model is not available. Please try again later.",
        )

    try:
        features = _build_features(payload)
        risk_score = float(model.predict_proba([features])[0][1])  # P(outbreak)
        confidence = float(model.predict_proba([features]).max())
        top_factors = _extract_top_factors(model, features)
    except Exception as exc:
        logger.exception(f"Model inference failed for region '{payload.region}': {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model inference failed. Please check your input and try again.",
        )

    response = PredictionResponse(
        prediction_id=f"pred_{uuid.uuid4().hex[:12]}",
        region=payload.region,
        risk_score=round(risk_score, 4),
        risk_level=_risk_level(risk_score),
        confidence=round(confidence, 4),
        top_factors=top_factors,
        predicted_at=datetime.now(timezone.utc).isoformat(),
    )

    logger.info(
        f"Prediction complete | region={response.region} "
        f"score={response.risk_score} level={response.risk_level}"
    )
    return response
