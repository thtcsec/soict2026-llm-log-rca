"""Unevaluated RCA interface mock for the SOICT 2026 design."""

import json
class RCANarrationDesignMock:
    """
    Builds the intended prompt and returns a schema placeholder.

    This class does not invoke an LLM, retrieve a runbook, infer a root cause, or
    measure latency. Its output must not be used as experimental evidence.
    """
    def __init__(self, model_name="Phi-3-mini-4k-instruct-INT4"):
        self.model_name = model_name

    def construct_prompt(self, anomalous_sequence, anomaly_score, context_metadata):
        prompt = f"""[SYSTEM PROMPT - AIOPS INCIDENT ANALYST]
You are an expert Security Operations & Network Reliability Engineer.
Analyze the following anomalous log sequence detected by TCN-Transformer Autoencoder.

=== INCIDENT METADATA ===
Anomaly Confidence Score: {anomaly_score:.4f}
Source Subnet / Campus Location: {context_metadata.get('location', 'HUFLIT Campus Main Switch')}
Affected Services: {context_metadata.get('services', 'Authentication & Core Gateway')}

=== ANOMALOUS LOG TEMPLATE SEQUENCE ===
{json.dumps(anomalous_sequence, indent=2)}

=== REQUIRED OUTPUT FORMAT (JSON) ===
1. Incident Severity (CRITICAL / HIGH / MEDIUM / LOW)
2. Root Cause Summary (1-2 sentences)
3. Technical Explanation & Attack Vector
4. Recommended Immediate Remediation Playbook (Step-by-step)
"""
        return prompt

    def generate_rca(self, anomalous_sequence, anomaly_score, context_metadata):
        prompt = self.construct_prompt(anomalous_sequence, anomaly_score, context_metadata)
        return {
            "evaluation_status": "DESIGN_ONLY_NOT_MODEL_OUTPUT",
            "prompt_preview": prompt,
            "severity": None,
            "root_cause_summary": None,
            "technical_explanation": None,
            "retrieved_evidence": [],
            "remediation_playbook": [],
            "inference_time_ms": None,
        }

if __name__ == "__main__":
    engine = RCANarrationDesignMock()
    dummy_seq = [
        "auth_service IP <IP> User guest login failed invalid password",
        "auth_service IP <IP> User guest login failed invalid password",
        "auth_service IP <IP> User admin login failed invalid password",
        "firewall_core Drop packet from <IP> port <NUM>"
    ]
    meta = {"location": "HUFLIT Campus Server Room", "services": "Active Directory / Radius"}
    result = engine.generate_rca(dummy_seq, 0.924, meta)
    print(json.dumps(result, indent=2))
