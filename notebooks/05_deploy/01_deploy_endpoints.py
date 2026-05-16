# Databricks notebook source
# MAGIC %md
# MAGIC # 05 / Deploy models to Mosaic AI Model Serving
# MAGIC
# MAGIC Creates (or updates) two CPU serving endpoints, one per model, pointing at the
# MAGIC `@champion` version. Inference tables are enabled via **AI Gateway** so we get a
# MAGIC Delta log of every scoring request — used by the monitoring step.

# COMMAND ----------

# MAGIC %run ../00_setup/config

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)
from mlflow import MlflowClient

mlflow_uc = MlflowClient(registry_uri="databricks-uc")
w = WorkspaceClient()

# COMMAND ----------

def _deployable_version(model_name: str) -> str:
    """Prefer @champion; fall back to @candidate if no champion has been promoted yet.
    Raises if neither alias is set."""
    for alias in ("champion", "candidate"):
        try:
            mv = mlflow_uc.get_model_version_by_alias(model_name, alias)
            print(f"   using {model_name}@{alias} (v{mv.version})")
            return mv.version
        except Exception:
            continue
    raise RuntimeError(f"No @champion or @candidate alias on {model_name}")


def _ai_gateway_inference_config(served_name: str):
    """Try to build an AI Gateway config with inference tables. Falls back to None
    if the SDK version doesn't have the right symbols (older SDKs will skip it)."""
    try:
        from databricks.sdk.service.serving import (
            AiGatewayConfig,
            AiGatewayInferenceTableConfig,
        )
        return AiGatewayConfig(
            inference_table_config=AiGatewayInferenceTableConfig(
                enabled=True,
                catalog_name=CATALOG,
                schema_name=SCHEMA,
                table_name_prefix=f"{served_name}_inference",
            )
        )
    except Exception as e:
        print(f"   AI Gateway inference table not configured (SDK skipped): {e}")
        return None


def _wait_until_not_updating(endpoint_name: str, timeout_s: int = 1200) -> None:
    """Block until the endpoint's config_update is NOT_UPDATING. Tolerant of NotFound."""
    import time
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            ep = w.serving_endpoints.get(endpoint_name)
            cu = ep.state.config_update if ep.state else None
            if cu is None or str(cu).endswith("NOT_UPDATING"):
                return
            print(f"   waiting for {endpoint_name} (config_update={cu})")
        except Exception as e:
            print(f"   endpoint not yet visible ({e})")
            return
        time.sleep(20)
    raise RuntimeError(f"Endpoint {endpoint_name} did not settle after {timeout_s}s")


def deploy_endpoint(endpoint_name: str, model_name: str, served_name: str) -> None:
    version = _deployable_version(model_name)
    served = ServedEntityInput(
        entity_name=model_name,
        entity_version=version,
        name=served_name,
        workload_size="Small",
        scale_to_zero_enabled=True,
    )

    existing = [e for e in w.serving_endpoints.list() if e.name == endpoint_name]
    if existing:
        _wait_until_not_updating(endpoint_name)
        print(f"updating endpoint {endpoint_name} -> {model_name} v{version}")
        w.serving_endpoints.update_config(name=endpoint_name, served_entities=[served])
    else:
        print(f"creating endpoint {endpoint_name} -> {model_name} v{version}")
        cfg = EndpointCoreConfigInput(name=endpoint_name, served_entities=[served])
        w.serving_endpoints.create(name=endpoint_name, config=cfg)

    # Apply AI Gateway inference table config as a separate call after the endpoint settles.
    ai_gateway = _ai_gateway_inference_config(served_name)
    if ai_gateway is not None:
        try:
            _wait_until_not_updating(endpoint_name)
            w.serving_endpoints.put_ai_gateway(
                name=endpoint_name,
                inference_table_config=ai_gateway.inference_table_config,
            )
            print(f"   AI Gateway inference tables enabled for {endpoint_name}")
        except Exception as e:
            print(f"   AI Gateway inference table config skipped: {e}")


CREDIT = f"{CATALOG}.{SCHEMA}.{CFG['credit_model_name']}"
FRAUD = f"{CATALOG}.{SCHEMA}.{CFG['fraud_model_name']}"

deploy_endpoint(CFG["credit_endpoint_name"], CREDIT, "credit_champion")
deploy_endpoint(CFG["fraud_endpoint_name"],  FRAUD,  "fraud_champion")

# COMMAND ----------

dbutils.notebook.exit("deploy_ok")
