from transformers import BertTokenizer

def remove_accents(text: str) -> str:
    accents_translation_table = str.maketrans(
    "áéíóúýàèìòùỳâêîôûŷäëïöüÿñÁÉÍÓÚÝÀÈÌÒÙỲÂÊÎÔÛŶÄËÏÖÜŸ",
    "aeiouyaeiouyaeiouyaeiouynAEIOUYAEIOUYAEIOUYAEIOUY"
    )
    return text.translate(accents_translation_table)

class Data:

    def __init__(self, predicate2id, subj_type2id, obj_type2id):
        self.predicate2id = predicate2id
        self.subj_type2id = subj_type2id
        self.obj_type2id = obj_type2id

        self.train_loader = []
        self.test_loader = []
        self.dev_loader = []

        # if os.path.exist()


    def summary(self):
        print("DATA SUMMARY START:")
        print("     Train Instance Number: %s" % (len(self.train_loader)))
        print("     Test  Instance Number: %s" % (len(self.test_loader)))
        print("     Dev   Instance Number: %s" % (len(self.dev_loader)))
        print("DATA SUMMARY END.")
        sys.stdout.flush()


    def generate_data(self, args, name="train"):
        tokenizer = BertTokenizer.from_pretrained(args.bert_dir, do_lower_case=False)

        assert os.path.exist(os.path.join(args.data_dir, 'train.json'))
        tmp = [json.loads(l) for l in tqdm(open(os.path.join(args.data_dir, 'train.json'), 'rb'))]
        tmp = data_progress(args, tmp, tokenizer)
        self.train_loader[tmp[i:i+batch_size] for i in range(0, len(tmp), self.batch_size)]

        if os.path.exist(os.path.join(args.data_dir, 'test.json')):
            tmp = [json.loads(l) for l in tqdm(open(args.data_dir + '/test.json', 'rb'))]
            self.test_loader = data_progress(args, tmp, tokenizer)

        if os.path.exist(os.path.join(args.data_dir, 'dev.json')):
            tmp = [json.loads(l) for l in tqdm(open(args.data_dir + '/dev.json', 'rb'))]
            self.dev_loader = data_progress(args, tmp, tokenizer)


def getTokenizer_forseq(tokenizer, lstseq):
    tokens = ['[CLS]']
    for token in lstseq:
        token = tokenizer.tokenize(token)
        for t in token:
            tokens.append(t)
        tokens.append('[unused1]')
    tokens.append('[SEP]')
    return tokens


def getTokenSPN(tokenizer, sentText):
    sentText = remove_accents(sentText)
    tokens = [tokenizer.cls_token] + tokenizer.tokenize(sentText) + [tokenizer.sep_token]
    return tokens


def data_progress(args, data, tokenizer):
    """ 我不知道这里是要干嘛，我只知道这里写的跟 shit 一样 """
    processed = []
    for d in data:
        text = d['sentText']
        text = text.split(' ')
        if len(text) > 100:
            continue

        tokens = getTokenizer_forseq(tokenizer, text)  # tokenize text
        if len(tokens) > 512:
            tokens = tokens[:512]

        tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
        rels_dict = {}
        rel_nums = len(self.predicate2id)
        tokens_rel = [0] * rel_nums
        for triple in d['relationMentions']:
            triple = (getTokenizer_forwords(tokenizer, triple['em1Text']), triple['label'], getTokenizer_forwords(tokenizer, triple['em2Text']))
            sub_start_idx = find_head_idx(tokens, triple[0])
            obj_start_idx = find_head_idx(tokens, triple[2])
            if sub_start_idx == -1 or obj_start_idx == -1:
                continue
            sub_end_idx = sub_start_idx+len(triple[0])-1
            obj_end_idx = obj_start_idx+len(triple[2])-1
            sub_type = 1
            obj_type = 1
            rel_id =  self.predicate2id[triple[1]]
            tokens_rel[rel_id] = 1
            if rel_id not in rels_dict:
                rels_dict[rel_id] = []
            rels_dict[rel_id].append((sub_start_idx,sub_end_idx,sub_type,obj_start_idx,obj_end_idx,obj_type))
        if rels_dict:
            rel_ids = list(rels_dict.keys())
            for rel_id in rel_ids:
                rel_triples = rels_dict[rel_id]
                items = {}
                subj_type = collections.defaultdict(list)
                obj_type = collections.defaultdict(list)
                for triple in rel_triples:
                    subj_key = (triple[0],triple[1])
                    subj_type[subj_key].append(triple[2])
                    if subj_key not in items:
                        items[subj_key] = []
                    items[subj_key].append((triple[3],triple[4]))
                    obj_type[(triple[3],triple[4])].append(triple[5])

                if items:
                    s1, s2 = [0] * len(tokens), [0] * len(tokens)
                    ts1, ts2 = [0] * len(tokens), [0] * len(tokens)

                    for j in items:
                        s1[j[0]] = 1
                        s2[j[1]] = 1
                        stp = choice(subj_type[j])
                        ts1[j[0]] = stp
                        ts2[j[1]] = stp
                    k1, k2 = choice(list(items.keys()))
                    o1, o2 = [0] * len(tokens), [0] * len(tokens)
                    to1, to2 = [0] * len(tokens), [0] * len(tokens)
                    distance_to_subj = get_positions(k1, k2, len(tokens))

                    for j in items[(k1, k2)]:
                        o1[j[0]] = 1
                        o2[j[1]] = 1
                        otp = choice(obj_type[(j[0], j[1])])
                        to1[j[0]] = otp
                        to2[j[1]] = otp
                    processed += [(tokens_ids, tokens_rel, [k1], [k2], s1, s2, o1, o2, ts1, ts2, to1, to2,[rel_id],
                                    distance_to_subj, [stp])]
    return processed

def get_nearest_start_position(S1):
    nearest_start_list = []
    current_distance_list = []
    for start_pos_list in S1:
        nearest_start_pos = []
        current_start_pos = 0
        current_pos = []
        flag = False
        for i, start_label in enumerate(start_pos_list):
            if start_label > 0:
                current_start_pos = i
                flag = True
            nearest_start_pos.append(current_start_pos)
            if flag > 0:
                if i-current_start_pos > 10:
                    current_pos.append(499)
                else:
                    current_pos.append(i-current_start_pos)
            else:
                current_pos.append(499)
        nearest_start_list.append(nearest_start_pos)
        current_distance_list.append(current_pos)
    return nearest_start_list, current_distance_list

def seq_padding(X):
    L = [len(x) for x in X]
    ML = max(L)
    return [x + [0] * (ML - len(x)) for x in X]

def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))


def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]

def getTokenizer_forseq(tokenizer, lstseq):
    tokens = ['[CLS]']
    for token in lstseq:
        token = tokenizer.tokenize(token)
        for t in token:
            tokens.append(t)
        tokens.append('[unused1]')
    tokens.append('[SEP]')
    return tokens

def getTokenizer_forwords(tokenizer, words):
    words = words.split(' ')
    tokens = []
    for token in words:
        token = tokenizer.tokenize(token)
        for t in token:
            tokens.append(t)
        tokens.append('[unused1]')
    return tokens

def find_head_idx(text, entity):
    entity_len = len(entity)
    for i in range(len(text)):
        if text[i: i + entity_len] == entity:
            return i
    return -1