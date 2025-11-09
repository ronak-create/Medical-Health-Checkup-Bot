from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes

pipeline = InferencePipeline.init(
    model_id="yolov12n-640",
    video_reference=0,
    on_prediction=render_boxes
)

pipeline.start()
pipeline.join()