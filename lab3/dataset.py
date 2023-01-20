import os
import random

from torch.utils.data import Dataset

from lab3.utils import read_file


class NERDataset(Dataset):

    def __init__(self, path: str):
        self.__path = path
        self.annos = self.__get_annos(path)

    def __get_annos(self, path: str):
        files = os.listdir(path)
        files = [file for file in files if '.ann' in file]
        return files

    def split_data(self, part: float = 0.8):
        train_count = int(self.__len__() * part)
        train_anno = random.sample(self.annos, train_count)
        test_anno = list(set(self.annos) - set(train_anno))
        train_data, test_data = NERDataset(self.__path), NERDataset(self.__path)
        train_data.annos, test_data.annos = train_anno, test_anno
        return train_data, test_data

    def __getitem__(self, item):
        anno = read_file(os.path.join(self.__path, self.annos[item]))
        full_text = anno.txt_data
        words = list()
        tags = list()
        for ner in anno.ners:
            s_pos, e_pos = ner[1:]
            words.append(full_text[s_pos: e_pos])
            tags.append(ner[0])

        words = tuple(words)
        tags = tuple(tags)

        return words, tags

    def get_re_item(self, item: int):
        anno = read_file(os.path.join(self.__path, self.annos[item]))
        full_text = anno.txt_data
        words = list()
        for ner in anno.relations:
            s_pos, e_pos = ner[1:]
            words.append(full_text[s_pos: e_pos])

        tags = list()
        sentences = list()
        for rel in anno.relations:
            first, second = rel[1:]
            sentences.append(words[first - 1])
            tags.append(rel[0])


        words = tuple(words)
        tags = tuple(tags)

        return words, tags

    def word_to_ix(self):
        word_to_ix = {}
        for sentence, tags in self:
            for word in sentence:
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)
        return word_to_ix

    def tag_to_ix(self):
        file = read_file(os.path.join(self.__path, self.annos[0]))
        ners = list(set([ner[0] for ner in file.ners]))
        tag_to_ix = {value: key for key, value in enumerate(ners)}
        tag_to_ix["<START>"] = len(tag_to_ix)
        tag_to_ix["<STOP>"] = len(tag_to_ix)
        return tag_to_ix

    def __len__(self):
        return len(self.annos)
