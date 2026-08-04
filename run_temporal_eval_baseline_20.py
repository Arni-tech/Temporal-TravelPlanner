from evaluation.temporal_feasibility import run_temporal_eval_block

run_temporal_eval_block(
    output_dir=r"C:\Users\negia\trip_plan\outputs_baseline",
    system_name="baseline",
    set_type="validation",
    model_name="gpt-3.5-turbo-0125",
    mode="two-stage",
    num_samples=20,
    save_dir=r"C:\Users\negia\trip_plan\temporal_eval_outputs_baseline_20",
)