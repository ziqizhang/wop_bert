import sys

from transformers import BertTokenizer, BertForSequenceClassification

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(sys.argv[1], do_lower_case=True)
    cachedir=None
    if len(sys.argv)>2:
        cachedir=sys.argv[2]
    model = BertForSequenceClassification.from_pretrained(
        sys.argv[1],  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=5,  # The number of output labels.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
        cache_dir=cachedir
    )
    print("success")