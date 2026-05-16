# Databricks notebook source
# MAGIC %md
# MAGIC # Shared configuration loader
# MAGIC
# MAGIC Every other notebook starts with `%run ../00_setup/config`. That call exposes:
# MAGIC
# MAGIC - `CFG` — dict of every key in `conf/config.yaml`
# MAGIC - `CATALOG`, `SCHEMA`, `VOLUME`
# MAGIC - `table(name)` — returns the 3-level UC name for a logical table key
# MAGIC - `widgets are created for catalog / schema / volume` so a job can override them
# MAGIC
# MAGIC Nothing else hardcodes catalog / schema — change `config.yaml` (or override the widgets)
# MAGIC and the whole pipeline retargets cleanly.

# COMMAND ----------

import os
import yaml

# Resolve the repo root by walking up from the notebook path.
def _repo_root() -> str:
    # On Databricks, dbutils.notebook.entry_point gives us the notebook path.
    try:
        nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()  # type: ignore # noqa
        # /Workspace/.../mlops_testing_end_to_end/notebooks/00_setup/config -> /Workspace/.../mlops_testing_end_to_end
        return "/Workspace" + nb_path.rsplit("/notebooks/", 1)[0]
    except Exception:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

REPO_ROOT = _repo_root()
CONFIG_PATH = os.path.join(REPO_ROOT, "conf", "config.yaml")

with open(CONFIG_PATH, "r") as f:
    CFG = yaml.safe_load(f)

# COMMAND ----------

# Make all key parameters overridable via widgets so a job can run against any target
dbutils.widgets.text("catalog", CFG["catalog"], "Unity Catalog")
dbutils.widgets.text("schema", CFG["schema"], "Schema")
dbutils.widgets.text("volume", CFG["volume"], "Volume")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME = dbutils.widgets.get("volume")

# COMMAND ----------

def table(name_key: str) -> str:
    """Return the fully-qualified UC table name for a logical key from config.yaml.

    >>> table("raw_applications")
    'mlops_lending.credit_fraud.raw_applications'
    """
    short = CFG["tables"][name_key]
    return f"{CATALOG}.{SCHEMA}.{short}"

def volume_path(*parts: str) -> str:
    return f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/" + "/".join(parts)

print(f"Configured for {CATALOG}.{SCHEMA} (volume={VOLUME})")
