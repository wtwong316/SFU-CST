# SFU-CST
A Novel Text-Based Chatbot Screening Tool for Mental Health Assessment with Generative AI

## Introduction
Chatbot Screening Tool (CST) is a text-based chatbot designed to collect answers to specific screening questionnaires for mental health. The system is designed to achieve three goals: task completion, effective communication, and relationship building. The system aims to facilitate the development of socially intelligent mental healthcare conversational agents, bringing us one step closer to providing effective, humanized assessments. Unlike rigid questionnaires, the CST engages users in a friendly conversation. It does not follow the exact order or wording of the questionnaire, making it more user-friendly. By building trust with users, the CST can go beyond the questionnaire’s limitations. It may uncover valuable information, such as the origins of symptoms, which are not captured by the questionnaire alone.  

## Methodology
CST employs a hybrid model that combines finite state machines with data-driven techniques and generative AI. This hybrid approach ensures accurate symptom detection while maintaining a conversational flow. To address time and space complexity challenges, we introduce dynamic priority queues.  These optimize the chatbot’s performance during interactions. We fine-tune ChatGLM’s 6B pre-trained LLM model to detect PHQ-9 symptoms effectively.

## Data Source
To generate training and testing data suitable for detecting PHQ-9 symptoms, we re-annotated the labels of a published dataset D4 that mimics dialogues between a doctor and a patient during depression assessment. You can obtain D4 dataset from the paper mentioned in the Acknowledgement. Our dataset will not be released until further notice.

## Experimental Results
Our method achieves good accuracy, precision, recall, and F1 scores (>97%) when a single symptom appears in the patient’s answer to the question symptom. 

## Limitation
Detecting multiple symptoms remains a challenge. Successful deployment in clinical settings requires further refinement and validation.

## Extensibility
### Other Mental Health Assessment tests
The developed system can extend and allowing for inclusion of General Anxiety Disorder-7 (GAD-7), General Health Questionnaire (GHQ) and Depression Anxiety Stress Scale (DASS-21) for the style of the questionnaires is similar to PHQ-9. We just need a set of the sample responses to fine-tune the pre-trained LLM to support the target assessment test.

### Other Dialects
Using translation tools, we have the capability to transform our dataset into various dialects. Looking ahead, with a sufficient number of native target speaking participants in our experiments, we will be able to evaluate the potential of CST in providing support for such dialect.

## Getting Started
!!! This project is under development. If everything runs well, it will be announced here.
### Environment Setup
#### Download this repository
```shell
git clone https://github.com/wtwong316/SFU-CST.git
cd SFU-CST
```
#### Setup the virtual environment
make sure you have python3.10 installed and can be accessed.
```shell
python3.10 -m venv ./venv
source venv/bin/activate
```
#### Install dependencies with pip
```shell
pip install -r requirments.txt
```
#### Run the program
Run the script
```shell
source run_script.sh
```

Or run the following command
```shell
python webdemo.py --do_predict --overwrite_cache --prompt_column prompt --response_column response --model_name_or_path THUDM/chatglm2-6b --ptuning_checkpoint ./depression-chatglm2-6b/checkpoint-3000 --output_dir ./output --overwrite_output_dir --max_source_length 256 --max_target_length 256 --per_device_eval_batch_size 1 --predict_with_generate --pre_seq_len 128 --quantization_bit 8 --local_rank -1 
```

#### Usage
Here is the sample of a session of doctor-patient dialogues in a conversation.

| Role | Dialogue |
| ----- | ----- |
| 医生 | 你有什么问题吗? |
| 病人 | 除非有人监督我多吃一点，否则我无法进食。|
| 医生 | 嗯嗯! 我明白了。在过去两周内，您是否经常因为吃饭情况而烦恼, 有几天、一半以上或几乎每天？|
| 病人 | 一半以上 |
| 医生 | 有没有对以前喜欢的事情不感兴趣呢？|
| 病人 | 我感觉最近对什么都提不起兴趣，都很无聊。|
| 医生 | 在过去两周内，您是否经常因为缺乏兴趣或乐趣而烦恼, 有几天、一半以上或几乎每天？|
| 病人 | 时常 |
| 医生 | 有没有觉得心情低落吗？|
| 病人 | 觉得自己特没用，什么事都做不好，感觉生活没有了盼头
| 医生 | 在过去两周内，您是否经常因为心情低落的情况而烦恼, 有几天、一半以上或几乎每天？ |
| 病人 | 将近每天 |
| 医生 | 会不会对自己没有信心或觉得自己没有价值吗？|
| 病人 | 就是总觉得自己做不好 没有信心 |
| 医生 | 在过去两周内，您是否感到自己没有信心或觉得自己没有价值的情况而烦恼, 有几天、一半以上或几乎每天？|
| 病人 | 有时候 |
| 医生 | 那最近睡眠还好吗？ 有没有难以入睡或者容易醒的情况呀? |
| 病人 | 睡眠挺好的，主要是太累了，很快就可以入睡 |
| 医生 | 有没有觉得比较疲倦或者疲惫吗？ |
| 病人 | 我快要考试了，但是我总是学一会儿就好累 思考问题也会反应不过来 |
| 医生 | 在过去两周内，您是否经常因为比较疲倦或者疲惫的情况而烦恼, 有几天、一半以上或几乎每天？|
| 病人 | 超过一半 |
| 医生 | 在过去两周内，您是否经常因为走神或做事时注意力集中不了的情况而烦恼, 有几天、一半以上或几乎每天？|
| 病人 | 有一段时间 |
| 医生 | 会不会走神或做事時注意力集中不了 |
| 病人 | 就是我做事情很容易分心 集中不了注意力 |
| 医生 | 在过去两周内，您是否经常因为走神或做事时注意力集中不了的情况而烦恼, 有几天、一半以上或几乎每天？|
| 病人 | 一半以上 |
| 医生 | 有没有感到焦躁, 烦躁不安或者反应很慢吗？|
| 病人 | 经常觉得很烦躁，反应也迟钝了 |
| 医生 | 在过去两周内，您是否感到焦躁, 烦躁不安或者反应很慢的情况而烦恼, 有几天、一半以上或几乎每天？ |
| 病人 | 将近每天 |
| 医生 | 是否有过自杀自残的想法呢？|
| 病人 | 沒有想過 |
| 医生 | 还有其他问题吗？|
| 病人 | 没有什么 |

### Web Demo
![web-demo](resources/CST-Screenshot.jpg)

## Authors
Data Science Research Centre for Social Policies and Service, Saint Francis University, Hong Kong (https://www.sfu.edu.hk/dsrc)

## Acknowledgement
The authors would like to express their gratitude to all study participants for their support, to SJTU X-LANCE Lab for granting approval to access the D4 dataset. Their paper can be accessed via DOI: https://doi.org/10.48550/arXiv.2205.11764 . 
This a related githhub site for the D4_baseline https://github.com/BigBinnie/D4_baseline.git.

## License

The code of this repository is licensed under [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0). The use of the CST model weights is subject to the [Model License](MODEL_LICENSE). ChatGLM2-6B weights are **completely open** for academic research.

## Citation

If you find our work useful, please consider citing the this github site in this moment.
