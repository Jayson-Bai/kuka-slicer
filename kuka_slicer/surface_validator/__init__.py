"""Read-only printability validation for mapped surface source paths."""

from .reporting import render_html_report, write_validation_reports
from .validator import (
    CheckResult,
    SurfaceValidationReport,
    ValidatorLimits,
    validate_surface_job,
)

__all__ = [
    "CheckResult",
    "SurfaceValidationReport",
    "ValidatorLimits",
    "render_html_report",
    "validate_surface_job",
    "write_validation_reports",
]
