from utilities.heapq import PriorityQueue, HeapPQEntry
from utilities.queue import ListQueue
import numpy as np
import string

doctor_confirm_symptom_dict = dict()
topic_dict = dict()


def is_topic_exist(topic_queue, symptom_topic):
    global topic_dict
    topic = topic_dict[symptom_topic]
    return topic_queue.exists(topic)


def is_topic_match(topic_in_queue, symptom_topic):
    return topic_in_queue == topic_dict[symptom_topic]


def build_dpq(states):
    dpq = PriorityQueue()
    for state in states:
        if '#' not in state[0][0]:
            pe = HeapPQEntry(state[0], state[1])
            dpq.insert(pe)

    return dpq


def build_topics(states):
    topics = dict()
    for topic in states:
        i = 0
        for item in topic:
            if isinstance(item, str):
                if 'Confirm_State' in item:
                    doctor_confirm_symptom_dict[item] = 1
                if i == 0:
                    topics[item] = PriorityQueue()
                else:
                    pe = HeapPQEntry(item, i)
                    topics[topic[0]].insert(pe)
            i += 1
    return topics


def build_conversation_match(pairs):
    for pair in pairs:
        topic_dict[pair[0]] = pair[1]


def build_doctor_messages(messages):
    msgs_dict = dict()
    for message in messages:
        terms = message[0].split(':')
        if terms[0][0] == '#':
            continue
        if terms[0] not in msgs_dict:
            msgs_dict[terms[0]] = dict()
        msgs_dict[terms[0]][terms[1]] = message[1]

    return msgs_dict


def get_symptoms(text, current_state):
    topics = list()
    if text is not None:
        items = text.split(',')
        for item in items:
            topics.append(item)
        # deduplicate identified symptoms from LLM
        topics = list(set(topics))
    if len(topics) > 1:
        sorted_topics = list()
        for topic in topics:
            state = get_state(topic)
            if state[0] == current_state:
                sorted_topics.insert(0, topic)
            else:
                sorted_topics.append(topic)
        return sorted_topics
    else:
        return topics


def get_state(symptom):
    symptom_clean = symptom.strip(string.punctuation)
    symptom_str = symptom_clean.split('#')
    topic_str = symptom_str[1].split(':')
    return symptom_str[0], topic_str[0], topic_str[1]


def is_doctor_confirming_symptom(topic):
    if topic in doctor_confirm_symptom_dict:
        return True
    else:
        return False


def is_symptom_topic_confirmed(topic):
    if 'Incidence' in topic:
        return True
    else:
        return False
