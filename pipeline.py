import logging
import time
import warnings
from importlib.metadata import version
from os import environ

from dotenv import load_dotenv
from kfp.compiler import Compiler
from kfp.dsl import pipeline
from kfp.dsl.component_factory import create_component_from_func
from kfp.registry import RegistryClient
from pydantic.dataclasses import dataclass

from data_prep.component_func import prep_data
from eval.component_func import eval_model
from onnx_optimize.component_func import onnx_optimize
from train.component_func import train_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Ignore KFP warnings we cannot control.
warnings.filterwarnings("ignore", category=SyntaxWarning, module="kfp.dsl.types.type_annotations")
warnings.filterwarnings("ignore", category=SyntaxWarning, module="kfp.dsl.for_loop")

PIPELINE_PATH = "pipeline.yaml"


@dataclass(frozen=True)
class Environment:
    kfp_repository: str
    staging_bucket: str
    prep_data_docker_uri: str
    train_model_docker_uri: str
    eval_model_docker_uri: str
    onnx_optimize_docker_uri: str


def compile_upload() -> None:
    start_time = time.time()
    logger.info("Starting weather model pipeline template creation...")

    load_dotenv()
    env = Environment(
        kfp_repository=environ["KFP_REPOSITORY"],
        staging_bucket=environ["STAGING_BUCKET"],
        prep_data_docker_uri=environ["PREP_DATA_DOCKER_URI"],
        train_model_docker_uri=environ["TRAIN_MODEL_DOCKER_URI"],
        eval_model_docker_uri=environ["EVAL_MODEL_DOCKER_URI"],
        onnx_optimize_docker_uri=environ["ONNX_OPTIMIZE_DOCKER_URI"],
    )

    @pipeline(  # type: ignore[misc]
        name="weather-model-pipeline",
        description="An example workflow outlining how to prepare data, train a model, and evaluate its performance.",
        pipeline_root=env.staging_bucket,
    )
    def weather_model_pipeline(
        data_bucket: str = "weather_imgs",
        random_seed: int = 42,
        train_ratio: float = 0.8,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
        epochs: int = 10,
    ) -> None:
        prep_data_task = create_component_from_func(func=prep_data, base_image=env.prep_data_docker_uri, install_kfp_package=False)(
            data_bucket=data_bucket,
            random_seed=random_seed,
            train_ratio=train_ratio,
            test_ratio=test_ratio,
            val_ratio=val_ratio,
        )

        train_model_task = create_component_from_func(func=train_model, base_image=env.train_model_docker_uri, install_kfp_package=False)(
            data_bucket=data_bucket,
            random_seed=random_seed,
            epochs=epochs,
            train_split_info=prep_data_task.outputs["train_split_info"],
            val_split_info=prep_data_task.outputs["val_split_info"],
        )

        create_component_from_func(func=eval_model, base_image=env.eval_model_docker_uri, install_kfp_package=False)(
            data_bucket=data_bucket,
            test_split_info=prep_data_task.outputs["test_split_info"],
            torch_model=train_model_task.outputs["torch_model"],
        )

        create_component_from_func(func=onnx_optimize, base_image=env.onnx_optimize_docker_uri, install_kfp_package=False)(
            onnx_with_transform_model=train_model_task.outputs["onnx_with_transform_model"]
        )

    logger.info("Compiling weather model pipeline...")
    Compiler().compile(weather_model_pipeline, package_path=PIPELINE_PATH)
    template_name, version_name = RegistryClient(host=env.kfp_repository).upload_pipeline(
        file_name=PIPELINE_PATH, tags=[version("ml-workflows"), "latest"]
    )
    logger.info(
        f"Successfully created and uploaded template: "
        f"{template_name}, version-{version_name}. Time taken: {round(time.time() - start_time, 2)} seconds."
    )


if __name__ == "__main__":
    compile_upload()
