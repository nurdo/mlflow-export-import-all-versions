"""
Export a registered model and all the experiment runs associated with its latest versions.
"""

import os
import click
import mlflow
from mlflow_export_import.common.http_client import HttpClient
from mlflow_export_import.common import filesystem as _filesystem
from mlflow_export_import.run.export_run import RunExporter
from mlflow_export_import import utils

class ModelExporter():
    def __init__(self, export_metadata_tags=False, notebook_formats=["SOURCE"], filesystem=None):
        self.fs = filesystem or _filesystem.get_filesystem()
        self.client = mlflow.tracking.MlflowClient()
        self.client2 = HttpClient("api/2.0/preview/mlflow")
        self.run_exporter = RunExporter(self.client, export_metadata_tags=export_metadata_tags, notebook_formats=notebook_formats, filesystem=filesystem)

    def _get_all_model_versions(self, model_name):
        url = f"model-versions/search?filter=name=%27{model_name}%27"
        response = self.client2.get(url)
        model_versions = response["model_versions"]
        page_token = response.get("next_page_token")
        while page_token:
            response = self.client2.get(url + f"&page_token={page_token}")
            model_versions += response["model_versions"]
            page_token = response.get("next_page_token")
        return model_versions

    def export_model(self, output_dir, model_name):
        path = os.path.join(output_dir,"model.json")
        model = self.client2.get(f"registered-models/get?name={model_name}")
        del model["registered_model"]["latest_versions"]
        model["registered_model"]["all_versions"] = self._get_all_model_versions(model_name)
        for v in model["registered_model"]["all_versions"]:
            run_id = v["run_id"]
            opath = os.path.join(output_dir,run_id)
            self.run_exporter.export_run(run_id, opath)
            opath = opath.replace("dbfs:","/dbfs")
            run = self.client.get_run(run_id)
            v["artifact_uri"] = run.info.artifact_uri
        utils.write_json_file(self.fs, path, model)

@click.command()
@click.option("--model", help="Registered model name.", required=True, type=str)
@click.option("--output-dir", help="Output directory.", required=True, type=str)

def main(model, output_dir): # pragma: no cover
    print("Options:")
    for k,v in locals().items():
        print(f"  {k}: {v}")
    exporter = ModelExporter()
    exporter.export_model(output_dir, model)

if __name__ == "__main__":
    main()
