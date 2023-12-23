# wikitext
import os
import random
import numpy
import pickle
import nltk.data
import unicodedata


def _get_next_sentence(sentence, next_sentence, paragraphs, p=1.1):
    if random.random() < p:
        is_next = 1
    else:
        # `paragraphs` is a list of lists of lists
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = 0
    return sentence, next_sentence, is_next

def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False

def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False

def _tokenize_chinese_chars(text):
    """Adds whitespace around any CJK character."""
    output = []
    for char in text:
        cp = ord(char)
        if _is_chinese_char(cp):
            output.append(" ")
            output.append(char)
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)

def _is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def _run_strip_accents(self, text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)

def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

def _run_split_on_punc(text):
    """Splits punctuation on a piece of text."""
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
        char = chars[i]
        if _is_punctuation(char):
            output.append([char])
            start_new_word = True
        else:
            if start_new_word:
                output.append([])
            start_new_word = False
            output[-1].append(char)
        i += 1

    return ["".join(x) for x in output]


def _clean_text(text, do_lower_case=False):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or _is_control(char):
            continue
        if _is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    text = _tokenize_chinese_chars("".join(output))
    orig_tokens = whitespace_tokenize(text)
    split_tokens = []
    for token in orig_tokens:
        if do_lower_case:
            token = token.lower()
            token = _run_strip_accents(token)
        split_tokens.extend(_run_split_on_punc(token))

    output_tokens = whitespace_tokenize(" ".join(split_tokens))
    return " ".join(output_tokens)


def _get_nsp_data_from_paragraph(paragraph, paragraphs, p=1.1):
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        sent_a, sent_b, is_next = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs, p=p)
        if len(sent_a.strip()) > 2 and len(sent_b.strip()) > 2:
            nsp_data_from_paragraph.append((_clean_text(sent_a), _clean_text(sent_b), is_next))
    return nsp_data_from_paragraph


# p: how much noise should be appended into the raw dataset
def clean_wikitext_to_file(corpus_wikitext, corpus_file, p=1.1):
    examples_train = clean_wikitext(corpus_wikitext)
    # with open(corpus_file, "wb") as f:
    #     pickle.dump(examples_train, f)
    # dataset = numpy.array(examples_train)[:,:2]
    # numpy.savetxt(corpus_file, dataset, delimiter="\t", fmt="%s", encoding="utf-8")
    with open(corpus_file, "w", encoding='utf-8') as f:
        for line in examples_train:
            f.write("\t".join([str(i) for i in list(line)]) + "\n")


def clean_wikitext(corpus_wikitext, p=1.1):
    with open(corpus_wikitext, 'r', encoding="utf8") as f:
        lines = f.readlines()
    return clean_wikitext_from_str(lines, p)


def clean_wikitext_from_str(lines, p=1.1):
    # Uppercase letters are converted to lowercase ones
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    paragraphs = [tokenizer.tokenize(line.strip())
                  for line in lines if len(tokenizer.tokenize(line.strip())) >= 2]
    random.shuffle(paragraphs)

    examples_train = []
    for paragraph in paragraphs:
        examples_train.extend(
            _get_nsp_data_from_paragraph(paragraph, paragraphs, p=p))

    return examples_train