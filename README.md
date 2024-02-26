# SFU-CST
A Novel Text-Based Chatbot Screening Tool for Mental Health Assessment with Generative AI

## Introduction
Chatbot Screening Tool (CST) is a text-based chatbot designed to collect answers to specific screening questionnaires for mental health. The system is designed to achieve three goals: task completion, effective communication, and relationship building. The system aims to facilitate the development of socially intelligent mental healthcare conversational agents, bringing us one step closer to providing effective, humanized assessments. Unlike rigid questionnaires, the CST engages users in a friendly conversation. It does not follow the exact order or wording of the questionnaire, making it more user-friendly. By building trust with users, the CST can go beyond the questionnaire’s limitations. It may uncover valuable information, such as the origins of symptoms, which are not captured by the questionnaire alone.  

## Methodology
CST employs a hybrid model that combines finite state machines with data-driven techniques and generative AI. This hybrid approach ensures accurate symptom detection while maintaining a conversational flow. To address time and space complexity challenges, we introduce dynamic priority queues.  These optimize the chatbot’s performance during interactions. We fine-tune ChatGLM’s 6B pre-trained LLM model to detect PHQ-9 symptoms effectively.

## Data Source
To generate training and testing data suitable for detecting PHQ-9 symptoms, we re-annotated the labels of a published dataset D4 that mimics dialogues between a doctor and a patient during depression assessment.  

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

### Environment Setup


### Usage


### Web Demo


## Authors
Data Science Research Centre for Social Policies and Service, Saint Francis University, Hong Kong (https://www.sfu.edu.hk/dsrc)

## Acknowledgement
The authors would like to express their gratitude to all study participants for their support, to SJTU X-LANCE Lab for granting approval to access the D4 dataset. Their paper can be accessed via DOI: https://doi.org/10.48550/arXiv.2205.11764 . 
This a related githhub site for the D4_baseline https://github.com/BigBinnie/D4_baseline.git.

## License

The code of this repository is licensed under [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0). The use of the CST model weights is subject to the [Model License](MODEL_LICENSE). ChatGLM2-6B weights are **completely open** for academic research.

## Citation

If you find our work useful, please consider citing the this github site in this moment.
