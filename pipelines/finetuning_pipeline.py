import os

import boto3
import sagemaker
import sagemaker.session
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

from sagemaker.workflow.pipeline_context import LocalPipelineSession

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))

#standard get_Session
def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline(
    raw_data_path,
    region,
    role=None,
    default_bucket=None,
    model_package_group_name="FineTunedModels", 
    pipeline_name="FineTunedModelsPipeline",
    base_job_prefix="FineTunedModelsJob"
):
    """Gets a SageMaker ML Pipeline instance working with on CustomerChurn data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    #sagemaker_session = get_session(region, default_bucket)
    sagemaker_session = LocalPipelineSession()
    
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    # Parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    
    training_instance_type = ParameterString(
        name="TrainingInstanceType", default_value="ml.m5.2xlarge"
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value="PendingManualApproval",  # ModelApprovalStatus can be set to a default of "Approved" if you don't want manual approval.
    )
    input_data = ParameterString(
        name="InputDataUrl",
        default_value=raw_data_path,  
    )

    # Processing step for training set preparation
    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type="ml.m5.2xlarge",
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-{model_package_group_name}-preprocess", 
        sagemaker_session=sagemaker_session,
        role=role,
    )
    step_dataset_preparation = ProcessingStep(
        name=model_package_group_name + "Process",  
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(source=PARENT_DIR + "/processors/dataset_preparation/requirements.txt", destination="/opt/ml/processing/input/code/dataset_preparation/")
    ],
        outputs=[
            ProcessingOutput(output_name="output", source="/opt/ml/processing/output")
        ],
        code= PARENT_DIR + "/processors/dataset_preparation/dataset_preprocess.py",
        job_arguments=["--input-data", input_data],
    )
    
    #data_prep_path = step_dataset_preparation.properties.ProcessingOutputConfig.Outputs["output"].S3Output.S3Uri.to_string()
    #print(f"data_prep_path:{str(data_prep_path)}")
    
    
    '''
    
    
    # Training step for generating model artifacts
    model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/CustomerChurnTrain"
    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",  # we are using the Sagemaker built in xgboost algorithm
        region=region,
        version="1.0-1",
        py_version="py3",
        instance_type="ml.m5.xlarge",
    )
    xgb_train = Estimator(
        image_uri=image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        output_path=model_path,
        base_job_name=f"{base_job_prefix}/CustomerChurn-train",
        sagemaker_session=sagemaker_session,
        role=role,
    )
    xgb_train.set_hyperparameters(
        objective="binary:logistic",
        num_round=50,
        max_depth=5,
        eta=0.2,
        gamma=4,
        min_child_weight=6,
        subsample=0.7,
        silent=0,
    )
    step_train = TrainingStep(
        name="CustomerChurnTrain",
        estimator=xgb_train,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )

    # Processing step for evaluation
    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type="ml.m5.xlarge",
        instance_count=1,
        base_job_name=f"{base_job_prefix}/script-CustomerChurn-eval",
        sagemaker_session=sagemaker_session,
        role=role,
    )
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    step_eval = ProcessingStep(
        name="CustomerChurnEval",
        processor=script_eval,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
        ],
        code=os.path.join(BASE_DIR, "evaluate.py"),
        property_files=[evaluation_report],
    )

    # Register model step that will be conditionally executed
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json",
        )
    )

    # Register model step that will be conditionally executed
    step_register = RegisterModel(
        name="CustomerChurnRegisterModel",
        estimator=xgb_train,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )

    # Condition step for evaluating model quality and branching execution
    cond_lte = ConditionGreaterThanOrEqualTo(  # You can change the condition here
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="binary_classification_metrics.accuracy.value",  # This should follow the structure of your report_dict defined in the evaluate.py file.
        ),
        right=0.8,  # You can change the threshold here
    )
    step_cond = ConditionStep(
        name="CustomerChurnAccuracyCond",
        conditions=[cond_lte],
        if_steps=[step_register],
        else_steps=[],
    )
    
    
    
    '''
    
    
    
    
    # Pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_count,
            training_instance_type,
            model_approval_status,
            input_data,
        ],
        steps=[step_dataset_preparation],
        sagemaker_session=sagemaker_session,
    )
    
    
    return pipeline