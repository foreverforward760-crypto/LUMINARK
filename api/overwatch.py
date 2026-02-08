"""
LUMINARK OVERWATCH - FastAPI Endpoints
Exposes Overwatch functionality via REST API
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from luminark_overwatch import (
    OverwatchEngine,
    SystemMetrics,
    MonitoredSystem,
    SystemStatus,
    AlertLevel
)

# Create router
router = APIRouter(prefix="/overwatch", tags=["overwatch"])

# Global Overwatch instance
overwatch = OverwatchEngine()


# ========================
# Request/Response Models
# ========================

class SystemRegistration(BaseModel):
    id: str
    name: str
    description: str = ""
    tags: List[str] = []


class MetricsUpdate(BaseModel):
    resource_health: float = 5.0
    connectivity: float = 5.0
    throughput: float = 5.0
    error_rate: float = 0.0
    coherence: float = 5.0
    alignment: float = 5.0
    adaptability: float = 5.0
    integrity: float = 5.0


class SystemResponse(BaseModel):
    id: str
    name: str
    status: str
    sap_stage: float
    inversion_level: int


class DiagnosisResponse(BaseModel):
    system_id: str
    system_name: str
    timestamp: str
    metrics: Dict[str, Any]
    sap_analysis: Dict[str, Any]
    inversion: Dict[str, Any]
    special_conditions: Dict[str, Any]
    status: str


class InterventionResponse(BaseModel):
    system_id: str
    type: str
    reason: str
    sap_diagnosis: str
    actions: List[str]
    urgency: str
    maat_validated: bool


# ========================
# Endpoints
# ========================

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "operational",
        "service": "LUMINARK Overwatch",
        "version": "1.0.0",
        "systems_monitored": len(overwatch.systems)
    }


@router.get("/overview")
async def get_overview():
    """Get overview of all monitored systems"""
    return overwatch.get_overview()


@router.post("/systems", response_model=SystemResponse)
async def register_system(registration: SystemRegistration):
    """Register a new system for monitoring"""
    try:
        system = overwatch.register_system(
            system_id=registration.id,
            name=registration.name,
            description=registration.description,
            tags=registration.tags
        )
        return SystemResponse(
            id=system.id,
            name=system.name,
            status=system.status.value,
            sap_stage=system.sap_stage,
            inversion_level=system.inversion_level
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/systems")
async def list_systems():
    """List all registered systems"""
    return [
        {
            "id": s.id,
            "name": s.name,
            "status": s.status.value,
            "sap_stage": s.sap_stage,
            "inversion_level": s.inversion_level,
            "last_update": s.last_update.isoformat()
        }
        for s in overwatch.systems.values()
    ]


@router.get("/systems/{system_id}")
async def get_system(system_id: str):
    """Get details for a specific system"""
    if system_id not in overwatch.systems:
        raise HTTPException(status_code=404, detail=f"System {system_id} not found")

    system = overwatch.systems[system_id]
    return {
        "id": system.id,
        "name": system.name,
        "description": system.description,
        "status": system.status.value,
        "sap_stage": system.sap_stage,
        "inversion_level": system.inversion_level,
        "tags": system.tags,
        "last_update": system.last_update.isoformat(),
        "metrics": {
            "physical_stability": system.metrics.physical_stability(),
            "conscious_stability": system.metrics.conscious_stability(),
            "resource_health": system.metrics.resource_health,
            "connectivity": system.metrics.connectivity,
            "throughput": system.metrics.throughput,
            "error_rate": system.metrics.error_rate,
            "coherence": system.metrics.coherence,
            "alignment": system.metrics.alignment,
            "adaptability": system.metrics.adaptability,
            "integrity": system.metrics.integrity
        }
    }


@router.put("/systems/{system_id}/metrics")
async def update_metrics(system_id: str, metrics: MetricsUpdate):
    """Update metrics for a system"""
    if system_id not in overwatch.systems:
        raise HTTPException(status_code=404, detail=f"System {system_id} not found")

    try:
        new_metrics = SystemMetrics(
            resource_health=metrics.resource_health,
            connectivity=metrics.connectivity,
            throughput=metrics.throughput,
            error_rate=metrics.error_rate,
            coherence=metrics.coherence,
            alignment=metrics.alignment,
            adaptability=metrics.adaptability,
            integrity=metrics.integrity
        )
        overwatch.update_system_metrics(system_id, new_metrics)
        return {"status": "updated", "system_id": system_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/systems/{system_id}")
async def unregister_system(system_id: str):
    """Unregister a system"""
    if overwatch.unregister_system(system_id):
        return {"status": "unregistered", "system_id": system_id}
    raise HTTPException(status_code=404, detail=f"System {system_id} not found")


@router.get("/systems/{system_id}/diagnose", response_model=DiagnosisResponse)
async def diagnose_system(system_id: str):
    """Run SAP diagnosis on a system"""
    if system_id not in overwatch.systems:
        raise HTTPException(status_code=404, detail=f"System {system_id} not found")

    try:
        diagnosis = overwatch.diagnose_system(system_id)
        return DiagnosisResponse(**diagnosis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/systems/{system_id}/intervene", response_model=InterventionResponse)
async def get_intervention(system_id: str):
    """Get prescribed intervention for a system"""
    if system_id not in overwatch.systems:
        raise HTTPException(status_code=404, detail=f"System {system_id} not found")

    try:
        intervention = overwatch.prescribe_intervention(system_id)
        return InterventionResponse(
            system_id=intervention.system_id,
            type=intervention.intervention_type.value,
            reason=intervention.reason,
            sap_diagnosis=intervention.sap_diagnosis,
            actions=intervention.recommended_actions,
            urgency=intervention.urgency.name,
            maat_validated=intervention.maat_validated
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/systems/{system_id}/report")
async def get_system_report(system_id: str):
    """Get comprehensive report for a system"""
    if system_id not in overwatch.systems:
        raise HTTPException(status_code=404, detail=f"System {system_id} not found")

    try:
        return overwatch.get_system_report(system_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def get_alerts(
    system_id: Optional[str] = None,
    level: Optional[str] = None,
    limit: int = 50
):
    """Get alerts, optionally filtered"""
    alert_level = None
    if level:
        try:
            alert_level = AlertLevel[level.upper()]
        except KeyError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid alert level. Must be one of: {[l.name for l in AlertLevel]}"
            )

    alerts = overwatch.get_alerts(system_id, alert_level, limit)
    return [
        {
            "system_id": a.system_id,
            "level": a.level.name,
            "title": a.title,
            "message": a.message,
            "sap_context": a.sap_context,
            "timestamp": a.timestamp.isoformat(),
            "acknowledged": a.acknowledged
        }
        for a in alerts
    ]


@router.post("/diagnose-all")
async def diagnose_all_systems():
    """Run diagnostics on all registered systems"""
    results = {}
    for system_id in overwatch.systems:
        try:
            results[system_id] = overwatch.diagnose_system(system_id)
        except Exception as e:
            results[system_id] = {"error": str(e)}
    return {
        "timestamp": overwatch.get_overview()["timestamp"],
        "systems_diagnosed": len(results),
        "results": results
    }


# ========================
# Response Validation API
# ========================

# Import response validators
try:
    from luminark_overwatch.analyzers.validator import validate_response, ValidationResult
    VALIDATORS_AVAILABLE = True
except ImportError:
    VALIDATORS_AVAILABLE = False


class ResponseValidationRequest(BaseModel):
    response_text: str
    user_intent: Optional[str] = None
    user_is_wrong: bool = False
    strict_mode: bool = False
    analyzers: Optional[List[str]] = None


@router.post("/validate")
async def validate_ai_response(request: ResponseValidationRequest):
    """
    Validate an AI response for quality issues.

    Runs all LUMINARK OVERWATCH analyzers:
    - personal_assumption: Detects unsolicited psychoanalysis
    - fake_empathy: Detects performative emotional language
    - sycophancy: Detects excessive agreement
    - hallucination: Detects fabrication signs

    Returns verdict (pass/warning/fail/critical), score, and detailed issues.
    """
    if not VALIDATORS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Response validators not available"
        )

    try:
        result = validate_response(
            response_text=request.response_text,
            user_intent=request.user_intent,
            user_is_wrong=request.user_is_wrong,
            strict_mode=request.strict_mode,
            analyzers=request.analyzers
        )
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate/quick")
async def quick_validate(request: ResponseValidationRequest):
    """
    Quick pass/fail validation check.
    Returns simple boolean result for fast validation.
    """
    if not VALIDATORS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Response validators not available"
        )

    try:
        result = validate_response(
            response_text=request.response_text,
            user_intent=request.user_intent,
            user_is_wrong=request.user_is_wrong,
            strict_mode=request.strict_mode,
            analyzers=request.analyzers
        )
        return {
            "passed": result.passed,
            "verdict": result.verdict.value,
            "score": result.overall_score,
            "issue_count": len(result.issues)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========================
# Partner Bundle Export
# ========================

try:
    from luminark_overwatch.export_bundle import create_partner_bundle
    BUNDLE_EXPORT_AVAILABLE = True
except ImportError:
    BUNDLE_EXPORT_AVAILABLE = False


@router.get("/export/bundle")
async def export_partner_bundle():
    """
    Export comprehensive partner bundle as ZIP.

    Returns ZIP containing:
    - README.txt: Bundle overview
    - systems_report.csv: All systems with SAP diagnostics
    - systems_report.json: Full JSON export
    - alerts.csv: Alert history
    - overwatch_summary.html: Visual summary
    """
    if not BUNDLE_EXPORT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Bundle export not available"
        )

    from fastapi.responses import Response

    try:
        overview = overwatch.get_overview()
        systems = overview.get("systems", [])
        alerts_data = overwatch.get_alerts(limit=100)
        alerts = [
            {
                "system_id": a.system_id,
                "level": a.level.name,
                "title": a.title,
                "message": a.message,
                "sap_context": a.sap_context,
                "timestamp": a.timestamp.isoformat(),
                "acknowledged": a.acknowledged
            }
            for a in alerts_data
        ]

        bundle_bytes = create_partner_bundle(overview, systems, alerts)

        return Response(
            content=bundle_bytes,
            media_type="application/zip",
            headers={
                "Content-Disposition": "attachment; filename=luminark_overwatch_bundle.zip"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========================
# Demo Data Initialization
# ========================

def init_demo_data():
    """Initialize demo systems for testing"""
    # Register demo systems
    overwatch.register_system(
        "api-gateway",
        "API Gateway",
        "Main API routing and authentication",
        ["production", "critical"]
    )
    overwatch.register_system(
        "ml-inference",
        "ML Inference Engine",
        "Machine learning model serving",
        ["production", "gpu"]
    )
    overwatch.register_system(
        "data-pipeline",
        "Data Pipeline",
        "ETL and data processing",
        ["batch", "analytics"]
    )

    # Set initial metrics
    overwatch.update_system_metrics("api-gateway", SystemMetrics(
        resource_health=8.5, connectivity=9.0, throughput=7.5, error_rate=0.2,
        coherence=8.0, alignment=8.5, adaptability=7.0, integrity=9.0
    ))
    overwatch.update_system_metrics("ml-inference", SystemMetrics(
        resource_health=8.0, connectivity=8.5, throughput=9.0, error_rate=0.5,
        coherence=3.5, alignment=4.0, adaptability=3.0, integrity=4.5
    ))
    overwatch.update_system_metrics("data-pipeline", SystemMetrics(
        resource_health=2.0, connectivity=3.0, throughput=1.5, error_rate=8.0,
        coherence=2.5, alignment=2.0, adaptability=2.0, integrity=3.0
    ))


# Initialize demo data on module load
init_demo_data()
