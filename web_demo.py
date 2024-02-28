import os
import sys
from datasets import Dataset
import gradio as gr
import time

from utilities.io_utilities import (read_states, read_topics_in_states, read_doctor_messages, read_conversation_match)
from utilities.utilities import get_symptoms
from utilities.utilities import (get_state, is_doctor_confirming_symptom, is_symptom_topic_confirmed, is_topic_exist,
                                 is_topic_match)

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
# from datasets import load_dataset
from ptuning.arguments import ModelArguments, DataTrainingArguments
from trainer_seq2seq import Seq2SeqTrainer
import logging
from queuelib.rrqueue import RoundRobinQueue
from queuelib.queue import FifoMemoryQueue

data_args = None
model_args = None
training_args = None
model = None
trainer = None
max_seq_length = 0
tokenizer = None
state_dpq = read_states()
states = read_topics_in_states()
read_conversation_match()
doctor_messages = read_doctor_messages()
response = ""
stale_state_1 = False
stale_state_2 = False
stale_state_3 = False
stale_state_4 = False
logger = logging.getLogger(__name__)

top_p_value = 0.25
temperature_value = 0.5
total_points = 0
anhedonia_points = 0
sleeping_disorder_points = 0
eating_disorder_points = 0
depression_mood_points = 0
comfort_rrq = None


def post_process(self, y):
    #global response, stale_state_1, stale_state_2, stale_state_3
    #if y is None:
    #    return []
    #for i, (message, response) in enumerate(y):
    #    y[i] = (
    #        None if message is None else mdtex2html.convert(message),
    #        None if response is None else mdtex2html.convert(response),
    #    )
    #current_state, cs_priority = state_dpq.peek_head()
    #if current_state != 'End' and states[current_state].peek_head()[0] == 'End':
    #    state_dpq.remove_min()
    #    current_state, cs_priority = state_dpq.peek_head()
    #topic = states[current_state].peek_head()[0]
    #if stale_state_1:
    #    response = "Sorry, I don't understand what you just say. \n" + doctor_messages[current_state][topic]
    #elif stale_state_2:
    #    response = "We have talked about this before. Now, \n" + doctor_messages[current_state][topic]
    #elif stale_state_3:
    #    response = "Sorry, internal error... \n" + doctor_messages[current_state][topic]
    #else:
    #    response = doctor_messages[current_state][topic]
    if y is None:
        history = "医生: 你有什么问题吗?"
        y = [(history, response)]

    return y


gr.Chatbot.postprocess = post_process


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def process_sub_state(state, topic_queue, topic, topic_value, text):
    global stale_state_3, total_points
    raise_state = None
    topic_in_queue = topic_queue.peek_head()[0]

    if topic_in_queue == 'Confirm_State' and topic == 'State_Confirmed':
        if state == 'Inferiority':
            raise_state = 'Suicide_Ideation'

    if is_topic_match(topic_in_queue, topic):
        if topic == 'Incidence':
            if topic_value.isnumeric():
                total_points += int(topic_value)
            else:
                print(
                    "Incorrect incidence point: state {} , topic_in_queue: {} , topic: f{}, topic_value: {}".format(
                        state, topic_in_queue, topic, topic_value))
                stale_state_3 = True
        elif topic == 'Confirm_State':
            if topic_value.isnumeric():
                if int(topic_value) == 0:
                    topic_queue.remove_min()
        topic_queue.remove_min()
    else:
        # handle the response with the later topic confirmed in the same state
        print("Incorrect match: state: {} , topic_in_queue: {} , topic: {}".format(state, topic_in_queue, topic))
        stale_state_3 = True

    return raise_state


def preprocess_function_eval(examples):
    global data_args
    # Get the column names for input/target.
    prompt_column = data_args.prompt_column
    response_column = data_args.response_column
    history_column = data_args.history_column
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    inputs, targets = [], []
    for i in range(len(examples[prompt_column])):
        if examples[prompt_column][i] and examples[response_column][i]:
            query = examples[prompt_column][i]
            history = examples[history_column][i] if history_column is not None else None
            prompt = tokenizer.build_prompt(query, history)
            #if history_column is None or len(examples[history_column][i]) == 0:
            #    prompt = query
            #else:
            #    prompt = ""
            #    history = examples[history_column][i]
            #    for turn_idx, (old_query, response) in enumerate(history):
            #        prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
            #    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
            inputs.append(prompt)
            targets.append(examples[response_column][i])

    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, truncation=True, padding=True)
    labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)

    if data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def predict(predict_dataset):
    global training_args, data_args

    predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict",
                                      max_length=max_seq_length, do_sample=True, top_p=top_p_value,
                                      temperature=temperature_value)
    if trainer.is_world_process_zero():
        predictions = tokenizer.batch_decode(
            predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        predictions = [pred.strip() for pred in predictions]

    return predictions[0]


def convert_data(question, answer):
    if '在过去两周内' in question:
        q_incidence = {"prompt": [
            "Instruction:- 以下是醫生與病人的對話, 病人與PHQ-9症狀的出現頻率密切相關嗎？ 如果不確定，請回答'Unknown：1'。否則請回答'State1#incidence'的出現頻率(從低到高),'State1#incidence:1', 'State1#incidence:2' 或 'State1#incidence:3'。\nInput:- {0}\n{1}\n"],
                       "response": ["State1#incidence:1"]}
        q_incidence["prompt"][0] = q_incidence["prompt"][0].format(question, answer)
        qna = q_incidence
    else:
        q_symptom = {"prompt": [
            "Instruction:- 以下是醫生與病人的對話, 病人與PHQ-9症狀極其相關嗎？如果不確定，請回答'Unknown：1'。 如果病人沒有'State1#Topic_1'的症狀，請回答'State1#Topic_1:0'。 如果病人沒有'State1#Topic_1'和'State2#Topic_2'的症狀，請回答'State1#Topic_1:0,State2#Topic_2:0'。 如果病人有'State1#Topic_1'的症狀，請回答'State1#Topic_1:1'。 如果病人有'State1#Topic_1'和'State2#Topic_2'的症狀，請回答'State1#Topic_1:1,State2#Topic_2:1'。 \nInput:- {0}\n{1}\n"],
                     "response": ["State1#Topic_1:1"]}
        q_symptom["prompt"][0] = q_symptom["prompt"][0].format(question, answer)
        qna = q_symptom
    predict_dataset = Dataset.from_dict(qna)
    with training_args.main_process_first(desc="prediction dataset map pre-processing"):
        predict_dataset = predict_dataset.map(
            preprocess_function_eval,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=predict_dataset.column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on prediction dataset",
        )
    return predict_dataset


def process_state(history, text):
    global response, stale_state_1, stale_state_2, stale_state_3, comfort_rrq
    stale_state_1 = stale_state_2 = stale_state_3 = False
    if len(history) == 1:
        predict_dataset = convert_data(history[-1][0], '病人: ' + text)
    else:
        state = state_dpq.peek_head()[0]
        topic = states[state].peek_head()[0]
        doctor_message = doctor_messages[state][topic]
        predict_dataset = convert_data('医生: ' + doctor_message, '病人: ' + text)
    result = predict(predict_dataset)
    history[-1][1] = '病人: ' + text
    current_state, cs_priority = state_dpq.peek_head()
    symptoms = get_symptoms(result, current_state)

    if symptoms is not None:
        for symptom in symptoms:
            symptom_state, symptom_topic, symptom_topic_value = get_state(symptom)
            if symptom_state in states:
                if is_topic_exist(states[symptom_state], symptom_topic):
                    if symptom_state == current_state:
                        # The symptom detected is the same as the current state
                        raise_state = process_sub_state(symptom_state, states[symptom_state], symptom_topic,
                                                        symptom_topic_value, text)
                        if raise_state is not None:
                            left_child_state, left_child_priority = state_dpq.peek_left_child(0)
                            right_child_state, right_child_priority = state_dpq.peek_right_child(0)
                            if left_child_priority < right_child_priority:
                                state_dpq.change_priority(raise_state, (cs_priority + left_child_priority)/2 - 1)
                            else:
                                state_dpq.change_priority(raise_state, (cs_priority + right_child_priority)/2 - 1)
                    else:
                        # The symptom detected is not the same as current state.
                        # It may be a single symptom or one of the detected symptoms
                        if current_state == 'Start' and states[current_state].peek_head()[0] == 'Q_Any_Topics':
                            left_child_state, left_child_priority = state_dpq.peek_left_child(0)
                            right_child_state, right_child_priority = state_dpq.peek_right_child(0)
                            if left_child_priority < right_child_priority:
                                state_dpq.change_priority(symptom_state, (cs_priority + left_child_priority)/2)
                            else:
                                state_dpq.change_priority(symptom_state, (cs_priority + right_child_priority)/2)
                            states[current_state].remove_min()
                            # In the beginning, it is in start state.
                            # if the patient's message confirm a topic in that state and that topic exists
                            # in the topic queue, them remove it.
                            if (states[symptom_state].peek_head()[0] == 'Confirm_State' and
                                    symptom_topic == 'Confirm_State' and symptom_topic_value == '1'):
                                states[symptom_state].remove_min()
                        else:
                            # need to handle the topic of different confirmed
                            if is_doctor_confirming_symptom(states[current_state].peek_head()[0]):
                                # if patient's response confirmed topic of other state
                                if is_symptom_topic_confirmed(symptom_topic):
                                    state_dpq.change_priority(symptom_state, cs_priority/2)
                                    if states[symptom_state].peek_head()[0] == 'Confirm_State':
                                        states[symptom_state].remove_min()
                                    else:
                                        states[symptom_state].remove(symptom_topic)
                            else:
                                #if is_symptom_topic_confirmed(symptom_topic):
                                left_child_state, left_child_priority = state_dpq.peek_left_child(0)
                                right_child_state, right_child_priority = state_dpq.peek_right_child(0)
                                if left_child_priority < right_child_priority:
                                    state_dpq.change_priority(symptom_state, (cs_priority + left_child_priority)/2)
                                else:
                                    state_dpq.change_priority(symptom_state, (cs_priority + right_child_priority)/2)
                else:
                    print("Topic {} not in State {}".format(symptom_topic, symptom_state))
                    stale_state_2 = True
            else:
                print("Invalid detected State {}".format(symptom_state))
                stale_state_3 = True
    else:
        print("Reply with non symptom cannot be handled...")
        stale_state_1 = True

    current_state, cs_priority = state_dpq.peek_head()
    if current_state != 'End' and states[current_state].peek_head()[0] == 'End':
        state_dpq.remove_min()
        current_state, cs_priority = state_dpq.peek_head()
    topic = states[current_state].peek_head()[0]
    if stale_state_1:
        response = "抱歉，我不明白你剛才說的話。 現在，請回答:\n" + doctor_messages[current_state][topic]
    elif stale_state_2:
        response = "我們之前已經討論過這個 {} 問題。 現在，請回答: \n".format(symptom_state) + doctor_messages[current_state][topic]
    elif stale_state_3:
        response = "抱歉，內部錯誤...\n" + doctor_messages[current_state][topic]
    else:
        if symptom_topic_value == '1' and symptom_topic == 'Confirm_State':
            response = '医生: ' + comfort_rrq.pop() + doctor_messages[current_state][topic]
        else:
            response = '医生: ' + doctor_messages[current_state][topic]

    return history, gr.update(value="", interactive=False)


def bot(history):
    history.append(["", ""])
    for character in response:
        history[-1][0] += character
        time.sleep(0.01)
        yield history


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []


def greeting():
    return "Type here..."


def clear_history():
    global response, stale_state_1, stale_state_2, stale_state_3, total_points, state_dpq, states
    stale_state_1 = stale_state_2 = stale_state_3 = False
    total_points = 0
    state_dpq = read_states()
    states = read_topics_in_states()
    response = '医生: ' + doctor_messages['Start']['Q_Any_Symptoms']



with gr.Blocks() as demo:

    gr.HTML("""<h1 align="center">Data Science Research Centre<br/>Saint Francis University<br/>Chatbot Screening Tool (CST) for mental health</h1>""")

    #chatbot = gr.Chatbot().style(height=600)
    chatbot = gr.Chatbot(height=700)
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                #user_input = gr.Textbox(show_label=False, placeholder=greeting(), lines=5).style(container=False)
                user_input = gr.Textbox(show_label=False, placeholder=greeting(), lines=5, container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            #max_length = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)
            #top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
            #temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

            gr_total = gr.Number(value=total_points, label="Score", interactive=False)
    txt_msg = submitBtn.click(process_state, [chatbot, user_input], [chatbot, user_input], queue=False).then(
        bot, chatbot, chatbot
    )
    txt_msg.then(lambda: gr.update(interactive=True), None, [user_input], queue=False)

    emptyBtn.click(clear_history, None, chatbot, queue=False)

    @gr.on(inputs=user_input, outputs=gr_total)
    def get_total_points(text):
        return total_points


def memory_queue_factory(priority):
    return FifoMemoryQueue()


def initial_comfort_rrq():
    global comfort_rrq
    comfort_rrq = RoundRobinQueue(qfactory=memory_queue_factory)
    comfort_rrq.push('嗯嗯! 我明白了。', 0)
    comfort_rrq.push('好的, 我了解了。', 1)
    comfort_rrq.push('好的', 2)
    comfort_rrq.push('明白', 3)
    comfort_rrq.push('啊，这样子', 4)


def main():
    global model, tokenizer, trainer, max_seq_length, model_args, training_args, data_args, comfort_rrq

    initial_comfort_rrq()
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    # The parsing argument routine in hf_argparser.py parse_args_into_dataclasses() seems incorrect, hardcode set here
    training_args.local_rank = -1
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    config.pre_seq_len = model_args.pre_seq_len
    config.prefix_projection = model_args.prefix_projection

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    if model_args.ptuning_checkpoint is not None:
        # Evaluation
        # Loading extra state dict of prefix encoder
        model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)
        prefix_state_dict = torch.load(os.path.join(model_args.ptuning_checkpoint, "pytorch_model.bin"))
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    else:
        model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)

    if model_args.quantization_bit is not None:
        print(f"Quantized to {model_args.quantization_bit} bit")
        model = model.quantize(model_args.quantization_bit)
    if model_args.pre_seq_len is not None:
        # P-tuning v2
        model = model.half()
        model.transformer.prefix_encoder.float()
    else:
        # Finetune
        model = model.float()

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
        padding=False
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )
    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,
        save_prefixencoder=model_args.pre_seq_len is not None
    )
    max_seq_length = data_args.max_source_length + data_args.max_target_length + 1

    demo.queue().launch(share=False, inbrowser=True, server_name="127.0.0.1", server_port=7860, debug=True, show_api=True)


if __name__ == "__main__":
    main()
