"""IMDSOC - Intelligent Medication Dispensing and Shipping Optimization Center.

Reference implementation of "A Reference Architecture for Regulation-Aware,
Cost-Optimized Pharmacy Fulfillment". Deterministic eligibility rules first
eliminate ineligible sites; a set-partitioning MILP with a dollar-valued
objective and chance constraints on transit time then selects shipment plans.
"""
from .chance import chance_filter
from .cost import CostBreakdown, CostModel, PolicyParameters
from .eligibility import EligibilityEngine, EligibilityResult
from .models import (Carrier, Lot, Network, Order, Packaging, Patient, Product, ServiceLevel,
                     Site, TempClass)
from .optimizer import solve_set_partitioning
from .orchestrator import Decision, IMDSOCConfig, Orchestrator, PlanningResult
from .plans import Plan, PlanGenerator
from .predictive import PredictorSuite

__version__ = "0.1.0"

__all__ = [
    "Carrier", "CostBreakdown", "CostModel", "Decision", "EligibilityEngine", "EligibilityResult",
    "IMDSOCConfig", "Lot", "Network", "Order", "Orchestrator", "Packaging", "Patient", "Plan",
    "PlanGenerator", "PlanningResult", "PolicyParameters", "PredictorSuite", "Product",
    "ServiceLevel", "Site", "TempClass", "chance_filter", "solve_set_partitioning",
]
