MODIFY THE FUNCTION IN THE FOLLOWING FILE AS SUCH TO FIX PRINTING ERRORS

./virtualenv/lib/python3.12/site-packages/transformers/trainer_callback.py


class PrinterCallback(TrainerCallback):
    """
    A bare [`TrainerCallback`] that just prints the logs.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero and control.should_log:
            print(logs)

