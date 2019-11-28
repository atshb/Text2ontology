class SinglePhraseEncoder()

    def __init__(self, tokenizer, seq_len=50):
        self.seq_len = seq_len
        self.tokenizer = tokenizer

    def __call__(self, x):
        # BPEでトークンに分割
        tokens = self.tokenizer.tokenize('[CLS]' + x + '[SEP]')
        # トークンをidに
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        # テンソル化 & 最大長にあわせてパディング
        tokens += [0] * (self.seq_len - len(tokens)))

        return torch.LongTensor(tokens), torch.LongTensor(ttypes)


class BertRnn():
    def __init__():
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.classifier = TwinRnnClassifier()

    def __call__()
