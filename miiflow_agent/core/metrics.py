"""Metrics and observability for LLM operations."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


class MetricType(Enum):
    """Types of metrics collected."""
    
    TOKEN_USAGE = "token_usage"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    REQUEST_COUNT = "request_count"


@dataclass
class TokenCount:
    """Token usage information."""
    
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def __add__(self, other: "TokenCount") -> "TokenCount":
        return TokenCount(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


@dataclass 
class UsageData:
    """Usage metrics for a single request."""
    
    provider: str
    model: str
    operation: str
    tokens: TokenCount
    latency_ms: float
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects and aggregates LLM metrics."""
    
    def __init__(self):
        self._usage_history: List[UsageData] = []
        self._aggregated_metrics: Dict[str, Any] = {}
    
    def record_usage(self, usage: UsageData) -> None:
        """Record usage data for a request."""
        self._usage_history.append(usage)
        self._update_aggregates(usage)
    
    def _update_aggregates(self, usage: UsageData) -> None:
        """Update aggregated metrics."""
        key = f"{usage.provider}:{usage.model}"
        
        if key not in self._aggregated_metrics:
            self._aggregated_metrics[key] = {
                "total_requests": 0,
                "successful_requests": 0,
                "total_tokens": TokenCount(),
                "total_latency_ms": 0.0,
                "error_count": 0,
            }
        
        metrics = self._aggregated_metrics[key]
        metrics["total_requests"] += 1
        
        if usage.success:
            metrics["successful_requests"] += 1
        else:
            metrics["error_count"] += 1
            
        metrics["total_tokens"] += usage.tokens
        metrics["total_latency_ms"] += usage.latency_ms
    
    def get_metrics(self, provider: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
        """Get aggregated metrics, optionally filtered by provider/model."""
        if provider or model:
            filtered = {}
            for key, metrics in self._aggregated_metrics.items():
                p, m = key.split(":", 1)
                if (not provider or p == provider) and (not model or m == model):
                    filtered[key] = metrics
            return filtered
        
        return self._aggregated_metrics.copy()
    
    def get_usage_history(self, limit: Optional[int] = None) -> List[UsageData]:
        """Get usage history, optionally limited."""
        if limit:
            return self._usage_history[-limit:]
        return self._usage_history.copy()
    
    def reset(self) -> None:
        """Reset all collected metrics."""
        self._usage_history.clear()
        self._aggregated_metrics.clear()


@dataclass
class LLMMetrics:
    """Metrics snapshot for LLM operations."""
    
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: TokenCount = field(default_factory=TokenCount)
    average_latency_ms: float = 0.0
    error_rate: float = 0.0
    
    @classmethod
    def from_collector(cls, collector: MetricsCollector) -> "LLMMetrics":
        """Create metrics snapshot from collector."""
        all_metrics = collector.get_metrics()
        
        total_requests = sum(m["total_requests"] for m in all_metrics.values())
        successful_requests = sum(m["successful_requests"] for m in all_metrics.values())
        failed_requests = sum(m["error_count"] for m in all_metrics.values())
        
        total_tokens = TokenCount()
        total_latency = 0.0
        
        for metrics in all_metrics.values():
            total_tokens += metrics["total_tokens"]
            total_latency += metrics["total_latency_ms"]
        
        return cls(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_tokens=total_tokens,
            average_latency_ms=total_latency / max(total_requests, 1),
            error_rate=failed_requests / max(total_requests, 1),
        )