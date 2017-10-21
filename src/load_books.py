from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import unidecode
import string
import numpy as np
import io
import os

import tensorflow as tf

def parse_file(file_name):
    data_set = []
    with open(file_name, 'r') as f:
        current_book = None
        for line in f:
            if line.startswith('_BOOK_TITLE_'):
                if current_book is not None: data_set.append(current_book)
                current_book = ""
            else:
                current_book += line
    return data_set

#train_books = parse_file("../datasets/CBTest/data/cbt_train.txt")
#validation_books = parse_file("../datasets/CBTest/data/cbt_valid.txt")
#test_books = parse_file("../datasets/CBTest/data/cbt_test.txt")[:1]

def load_training_data(directory):

    books_dirs = []
    books_dirs.append(directory + "/ohenry/")
    books_dirs.append(directory + "/misc_stories/")
    books_dirs.append(directory + "/aesop/")
    #books_dirs.append(directory + "/shakespeare/")

    train_books = []
    for books_dir in books_dirs:
        lookup_path = books_dir + "*"
        print(lookup_path)
        books = tf.train.match_filenames_once(lookup_path)

        queue = tf.train.string_input_producer(books, num_epochs=1)

        reader = tf.WholeFileReader()
        _, bookcontents = reader.read(queue)

        init_op = (tf.global_variables_initializer(),
            # NEED LOCAL INIT TOO FOR MATCH_FILENAMES_ONCE GRRRRRRRR
            tf.local_variables_initializer())

        with tf.Session() as sess:
            sess.run(init_op)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            try: # this is the pattern Google recommends...
                while not coord.should_stop():
                    contents = sess.run(bookcontents)
                    train_books.append(contents.decode("utf-8"))
            except tf.errors.OutOfRangeError:
                print("Loaded all books")
            finally:
                coord.request_stop()
            coord.join(threads)
#        for book in os.listdir(books_dir):
#        for book in cloudstorage.listbucket(books_dir):
#            with io.open(books_dir + book, 'r') as f:
#            with cloudstorage.open(books_dir + book, 'r') as f:
#                try:
#                    train_books.append(f.read())
#                except BaseException as e:
#                    print("Error with book:", f.name)
#                    raise e

    seq = ("aaaaaaabaaaaaaabaaaaaaabaaaaaaabaaaaaaabaaaaaaabaaaaaaabaaaaaaab"
           "aaaaaaabaaaaaaabaaaaaaabaaaaaaabaaaaaaabaaaaaaabaaaaaaab")

    seq2 = ("aaaaaaabaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab")

    seq3 = ("abcaaaabaaaaaaaaaaaaaaacaaaaaaaaaaaaaaab")

    seq4 = ("aaaab")

    train_books = [book for book in train_books if (len(book) < 16000)]

    print("Num books is: ", len(train_books))

    # For testing, lets just reduce the total paramters by removing accents
    train_books = [unidecode.unidecode(book) for book in train_books]
    import re # Remove control characters except new lines

    train_books = [re.sub(r"\r\n", "\n", book) for book in train_books] #dos2unix
    # replace all non-tab or newline control characters with space
    train_books = [re.sub(r"[\x00-\x08]|[\x0B-\x1F]|\x7f", " ", book) for book in train_books]

    character_set = set()

    character_set.update(string.printable)
    character_set.remove('\r') # Don't care about carriage return, just use newline (line feed)
    character_set.remove('\x0b') # Don't care about vertical tab
    character_set.remove('\x0c') # Don't care about form feed (new page)

    assert all((character_set.contains(c) for c in book) for book in train_books)

    char_indices = { character : c_idx for c_idx, character in enumerate(sorted(character_set)) }
    index_chars = { c_idx : character for character, c_idx in char_indices.items() }

    print(sorted(char_indices.items()), len(character_set))

    class Books:
        def __init__(self):
            self.train_books = train_books
            self.character_set = character_set
            self.char_indices = char_indices
            self.index_chars = index_chars

        def logits(self, str, int_type):
            assert len(self.char_indices) <= np.iinfo(int_type).max
            return np.array(list(self.char_indices[char] for char in str), dtype=int_type)

        def one_hot(self, str):
            ret = np.zeros([len(str), len(self.character_set)])
            for idx, char in enumerate(str):
                if char != 0: ret[idx, self.char_indices[char]] = 1
            return ret

    return Books()
