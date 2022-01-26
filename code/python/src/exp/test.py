import sys

from transformers import BertTokenizer

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(sys.argv[1], do_lower_case=True)
    print("success")