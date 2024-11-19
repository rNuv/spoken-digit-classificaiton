import numpy as np

class SpeechToken:
    """
    class to encapsulate one utterance of a digit
    """

    def __init__(self, digit, token_index, mfccs, speaker_gender):
        """
        :param digit: digit being represented by SpeechToken
        :param token_index: block number within digit (1-660)
        :param mfccs: block of mfccs representing utterance
        :param speaker_gender: gender of speaker
        """

        self.digit = digit
        self.token_index = token_index
        self.mfccs = mfccs
        self.speaker_gender = speaker_gender
        self.analysis_windows = len(mfccs)


def read_data(file_path):
    """
    :param file_path: file path to data
    :return list of blocks where each block is a list of lists representing analysis frames of an utterance of a digit 
    """

    blocks = []
    block = []
    with open(file_path, "r") as file:
        for line in file:
            if line.strip():
                block.append(list(map(float, line.strip().split())))
            else:
                if block:
                    blocks.append(block)
                    block = []

    if block:
        blocks.append(block)
    return blocks


def create_tokens(blocks):
    """
    :param: blocks representing parsed data
    :return list of SpeechTokens
    """

    tokens = []
    num_blocks_per_digit = len(blocks) / 10
    num_blocks_per_gender = num_blocks_per_digit / 2

    for i, block in enumerate(blocks):
        digit = int(i // num_blocks_per_digit)
        if i % num_blocks_per_digit < num_blocks_per_gender:
            speaker_gender = "male"
        else:
            speaker_gender = "female"

        token_index = (i % num_blocks_per_digit) + 1
        tokens.append(SpeechToken(digit, token_index, np.array(block), speaker_gender))

    return tokens

def get_data(isTesting=False):
    """
    :param isTesting: specify whether to get testing or training data
    :return list of tokens representing either training or testing data
    """

    file_path = "data/Train_Arabic_Digit.txt"
    if isTesting:
        file_path = "data/Test_Arabic_Digit.txt"
    blocks = read_data(file_path)
    return create_tokens(blocks)

def extract_mfccs(tokens, digit):
    """
    :param tokens: list of SpeechToken objects representing an utterance
    :param digit: specific digit we want to extract
    :return (N, 13) np array where N is the total number of frames in the dataset for a specified digit
    """
    mfcc_frames = []
    for token in tokens:
        if token.digit == digit:
            for mfcc_coeffs in token.mfccs:
                mfcc_frames.append(mfcc_coeffs)
    mfcc_frames = np.asarray(mfcc_frames)
    return mfcc_frames

def extract_mfccs_by_gender(tokens, digit):
    male = []
    female = []
    for token in tokens:
        if token.digit == digit and token.speaker_gender == "male":
            for mfcc_coeffs in token.mfccs:
                male.append(mfcc_coeffs)
        if token.digit == digit and token.speaker_gender == "female":
            for mfcc_coeffs in token.mfccs:
                female.append(mfcc_coeffs)
    male = np.asarray(male)
    female = np.asarray(female)
    return male, female

if __name__ == '__main__':
    training_data = get_data()
    male, female = extract_mfccs_by_gender(training_data, 5)
    total = extract_mfccs(training_data, 5)