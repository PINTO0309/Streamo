from transformers.trainer_callback import TrainerControl, TrainerState

from swift.trainers.arguments import TrainingArguments
from swift.trainers.callback import DefaultFlowCallbackNew


def test_save_when_training_stops_before_max_steps(tmp_path):
    args = TrainingArguments(
        output_dir=str(tmp_path),
        save_strategy='steps',
        save_steps=100,
        eval_strategy='no',
        report_to=[],
    )
    state = TrainerState(global_step=1, max_steps=10, epoch=1)
    control = TrainerControl(should_training_stop=True)

    control = DefaultFlowCallbackNew().on_epoch_end(args, state, control)

    assert control.should_save is True
    assert control.should_evaluate is False


def test_skip_final_save_when_checkpoint_already_exists(tmp_path):
    (tmp_path / 'checkpoint-10').mkdir()
    args = TrainingArguments(
        output_dir=str(tmp_path),
        save_strategy='steps',
        save_steps=100,
        eval_strategy='no',
        report_to=[],
    )
    state = TrainerState(global_step=10, max_steps=10, epoch=1)
    state.last_model_checkpoint = str(tmp_path / 'checkpoint-10')
    control = TrainerControl(should_training_stop=True)

    control = DefaultFlowCallbackNew().on_epoch_end(args, state, control)

    assert control.should_save is False
