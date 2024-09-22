from kfp.dsl import Input, Metrics, Model, Output


def onnx_optimize(
    onnx_with_transform_model: Input[Model], optimization_metrics: Output[Metrics], optimized_onnx_with_transform_model: Output[Model]
) -> None:
    import logging
    import time

    import onnxruntime as rt

    start_time = time.time()
    logging.info("Started ONNX model optimization task.")

    sess_options = rt.SessionOptions()
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.optimized_model_filepath = optimized_onnx_with_transform_model.path
    rt.InferenceSession(f"{onnx_with_transform_model.path}.onnx", sess_options)
    optimized_onnx_with_transform_model.framework = (
        f"onnxruntime-{rt.__version__}, graphOptimizationLevel-{str(sess_options.graph_optimization_level)}"
    )
    optimization_metrics.log_metric("timeTakenSeconds", round(time.time() - start_time, 2))
    logging.info("Successfully finished ONNX model optimization task.")
