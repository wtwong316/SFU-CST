import pandas as pd
from utilities.utilities import (build_dpq, build_topics, build_conversation_match,
                                 build_doctor_messages)


def read_states():
    df = pd.read_csv('state_info/q_items.csv', header=None)
    return build_dpq(df.values)


def read_topics_in_states():
    df = pd.read_csv('state_info/q_sub_items.csv', header=None, names=range(10))
    return build_topics(df.values)


def read_conversation_match():
    df = pd.read_csv('state_info/conversation_match.csv', header=None)
    return build_conversation_match(df.values)


def read_doctor_messages():
    df = pd.read_csv('state_info/doctor_messages.csv', header=None, sep="#")
    return build_doctor_messages(df.values)
