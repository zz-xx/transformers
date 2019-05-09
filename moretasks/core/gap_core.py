class InputExample(object):

    def __init__(self, guid, text,
                 span_pronoun, span_a, span_b,
                 label=None):
        self.guid = guid
        self.text = text
        self.span_pronoun = span_pronoun
        self.span_a = span_a
        self.span_b = span_b
        self.label = label

    def new(self, **new_kwargs):
        kwargs = {
            "guid": self.guid,
            "text": self.text,
            "span_pronoun": self.span_pronoun,
            "span_a": self.span_a,
            "span_b": self.span_b,
            "label": self.label,
        }
        for k, v in new_kwargs.items():
            kwargs[k] = v
        return self.__class__(**kwargs)


class TokenizedExample(object):
    def __init__(self, guid, tokens,
                 span_pronoun, span_a, span_b,
                 label):
        self.guid = guid
        self.tokens = tokens
        self.span_pronoun = span_pronoun
        self.span_a = span_a
        self.span_b = span_b
        self.label = label

    def new(self, **new_kwargs):
        kwargs = {
            "guid": self.guid,
            "tokens": self.tokens,
            "span_pronoun": self.span_pronoun,
            "span_a": self.span_a,
            "span_b": self.span_b,
            "label": self.label,
        }
        for k, v in new_kwargs.items():
            kwargs[k] = v
        return self.__class__(**kwargs)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, guid, input_ids, input_mask, segment_ids,
                 span_pronoun, span_a, span_b,
                 label_id, tokens):
        self.guid = guid
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

        self.span_pronoun = span_pronoun
        self.span_a = span_a
        self.span_b = span_b

        self.label_id = label_id
        self.tokens = tokens

    def new(self, **new_kwargs):
        kwargs = {
            "guid": self.guid,
            "input_ids": self.input_ids,
            "input_mask": self.input_mask,
            "segment_ids": self.segment_ids,
            "span_pronoun": self.span_pronoun,
            "span_a": self.span_a,
            "span_b": self.span_b,
            "label_id": self.label_id,
            "tokens": self.tokens,
        }
        for k, v in new_kwargs.items():
            kwargs[k] = v
        return self.__class__(**kwargs)


class Batch:
    def __init__(self, input_ids, input_mask, segment_ids,
                 span_pronoun, span_a, span_b,
                 label_ids, tokens):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

        self.span_pronoun = span_pronoun
        self.span_a = span_a
        self.span_b = span_b

        self.label_ids = label_ids
        self.tokens = tokens

    def to(self, device):
        return Batch(
            input_ids=self.input_ids.to(device),
            input_mask=self.input_mask.to(device),
            segment_ids=self.segment_ids.to(device),
            span_pronoun=self.span_pronoun.to(device),
            span_a=self.span_a.to(device),
            span_b=self.span_b.to(device),
            label_ids=self.label_ids.to(device),
            tokens=self.tokens,
        )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, key):
        return Batch(
            input_ids=self.input_ids[key],
            input_mask=self.input_mask[key],
            segment_ids=self.segment_ids[key],
            span_pronoun=self.span_pronoun[key],
            span_a=self.span_a[key],
            span_b=self.span_b[key],
            label_ids=self.label_ids[key],
            tokens=self.tokens[key],
        )
