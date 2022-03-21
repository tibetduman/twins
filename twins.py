import multiprocessing as mp
import time
import scipy
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA = [] #usable data in numpy_array format
CATEGORY_LIST = [set() for i in range(50)] #list of categories, sets that contain company names
CATEGORY_NUM = {} #dictionary of CATEGORY_LIST set indices with category names
COMPANIES = {} #dictionary of companies by name as keys and short descriptions as values
BEST_K = 10 #the desired amount of most relevant companies
SENTENCE_LIMIT = 1000 #limit on how many sentences to use BERT on
LOWEST_SENTENCE_LENGTH = 8 #lower bound on the length of a description for a sentence
WORDS_TO_IGNORE = {"and", "a", "to", "the", ".", ",", "that", "which", "are", "is", "for",
"one", "of", "on", "or", "with", "their"} #may want more or less words in this list
DESCRIPTION_LIMIT = 40 #limit on number of words to check for similarities in heuristics
model = SentenceTransformer('bert-base-nli-mean-tokens') #NLP model to use

def data_wrangling():
    """Function to handle data wrangling to make raw excel file processable."""
    print("Starting data wrangling")
    data = pd.read_excel("dataset.xls")
    data = data.drop(["uuid", "uuid_1"], axis=1)
    data = data.to_numpy(na_value="none")
    for entry in data:
        valid_company = add_company(entry[0], entry[4]) #entry[1] is short, entry[4] is long description
        #entry[2], category_list, is ignored since less comprehensive than category_group_list
        if not valid_company:
            continue
        categories = seperate_by_commas_trim(entry[3])
        for category in categories:
            if category not in CATEGORY_NUM:
                category = category.lower()
                CATEGORY_NUM[category] = len(CATEGORY_NUM)
            CATEGORY_LIST[CATEGORY_NUM[category]].add(entry[0])

def add_company(name, description):
    """Function to add a company to the dictionary of companies, stores a mapping
    from name to short_description. Returns true if succeeds, false otherwise."""
    if len(description) > LOWEST_SENTENCE_LENGTH:
        COMPANIES[name] = description.lower()
        return True
    else:
        #description not long enough to be considered
        return False

def get_category(company_name):
    """Given a company name finds it's category(s) and returns them."""
    categories = []
    for category in CATEGORY_NUM:
        if company_name in CATEGORY_LIST[CATEGORY_NUM[category]]:
            categories.append(category)
    return categories

def seperate_by_commas_trim(sentence):
    """Function seperate a sentence by using comma as a delimiter and strips each subsentence,
    returns the resulting list of strings as a list."""
    sentence = sentence.lower()
    sentence = sentence.split(",")
    for s in sentence:
        s.strip()
    return sentence

def category_analysis():
    """Prints out useful statistics about categories of companies."""
    for category in CATEGORY_NUM:
        print("Category, {0}, has {1} many companies".format(category, len(CATEGORY_LIST[CATEGORY_NUM[category]])))
        #print_example_companies_from_category(category) #this is verbose
    print("there are {0} many companies".format(len(COMPANIES)))
    print("there are {0} many categories".format(len(CATEGORY_NUM)))

def sentences_in_category(categories):
    """Given a list of categories returns the descriptions that are associated with
    companies that are in the given categories."""
    descriptions_in_category = []
    for category in categories:
        for company in CATEGORY_LIST[CATEGORY_NUM[category]]:
            descriptions_in_category.append(COMPANIES[company])
    return descriptions_in_category

def split_into_n_sublists(lst, num):
    """Given a list lst, splits lst into num sublists of close size and returns the sublists in a list."""
    size = int(len(lst) / num)
    sublists = [lst[i:i + size] for i in range(0, len(lst), size)]
    return sublists

def closest_k_sentences(short_description, corpus_sentences):
    """Returns the k closest companies that have the best affiliated sentences with
    the given short description."""
    corpus = model.encode(corpus_sentences)
    query = model.encode(short_description)
    distances = scipy.spatial.distance.cdist([query], corpus, "cosine")[0]
    scores = zip(range(len(distances)), distances)
    print(len(corpus_sentences))
    return scores

def filter_sentences(sentence_list):
    """Given a sentence list returns the sentence list in descending order of it's sentences
    Jaccard similarity score."""
    sentence_with_similarity_score = []
    for sentence in sentence_list:
        sentence_with_similarity_score.append((sentence, jaccard_similarity(input_sentence, sentence)))
    sentence_with_similarity_score = sorted(sentence_with_similarity_score, key=lambda x: x[1])
    return sentence_with_similarity_score

def intersection(sentence1, sentence2):
    """Given two sentences as list of words returns how many words are shared."""
    ist = 0
    for word in sentence1:
        if word in sentence2:
            ist += 1
    return ist

def jaccard_similarity(sentence1, sentence2):
    """Given two sentences returns their Jaccard similarity score for simplified versions."""
    sentence1 = simplify_sentence(sentence1)
    sentence2 = simplify_sentence(sentence2)
    return intersection(sentence1, sentence2) / (len(sentence1) + len(sentence2))

def dice_similarity(sentence1, sentence2):
    """Given two sentences return their Dice Coefficient similarity score for simplified versions."""
    sentence1 = simplify_sentence(sentence1)
    sentence2 = simplify_sentence(sentence2)
    return 2 * intersection(sentence1, sentence2) / ((len(sentence1) + len(sentence2)))

def simplify_sentence(sentence):
    """Given a sentence returns the simplification of that sentence by removing
    string literals that don't have meaning on their own."""
    words = sentence.split()
    simple_sentence = [word for word in words if word not in WORDS_TO_IGNORE]
    return simple_sentence[0:min(DESCRIPTION_LIMIT, len(simple_sentence))]

def get_company(description, category):
    """Given a description and category returns the company name assosiciated with it."""
    for company in CATEGORY_LIST[CATEGORY_NUM[category]]:
        if description == COMPANIES[company]:
            return company
    print("Could not find the company")
    return "None"

if __name__ == '__main__':
    data_wrangling()
    category_analysis()
    start = time.perf_counter()
    number_of_processors = mp.cpu_count()
    print("Number of processors:", number_of_processors)
    input_sentence = "Zippin has developed the next generation of checkout-free technology enabling retailers to quickly deploy frictionless shopping in their stores. Our patent-pending approach uses AI, machine learning and sensor fusion technology to create the best consumer experience: banishing checkout lines and self-scanners for good, and letting shoppers zip in and out with their purchases. Zippin‚Äôs platform leverages product and shopper tracking through overhead cameras, as well as smart shelf sensors, for the highest level of accuracy even in crowded stores. Founded by industry veterans from Amazon and SRI with deep backgrounds in retail technology, AI and computer vision, Zippin is headquartered in San Francisco and has raised venture funding from Maven Ventures, Core Ventures Group, Pear Ventures, Expansion VC, and Montage Ventures.  For more information, visit www.getzippin.com."
    all_sentences = filter_sentences(sentences_in_category(["artificial intelligence"]))
    #all_sentences = sentences_in_category(["payments"]) #uncomment if u want no filtering
    sentences = [sentence[0] for sentence in all_sentences[0:SENTENCE_LIMIT]]
    print("Will run bert on {} many sentences.".format(len(sentences)))
    pool = mp.Pool(number_of_processors)
    optimized_sentences = split_into_n_sublists(sentences, number_of_processors)
    for i in range(number_of_processors):
        results = pool.apply_async(closest_k_sentences, args=(input_sentence, optimized_sentences[i]))
    pool.close()
    pool.join()
    result = results.get()
    #result = closest_k_sentences(sentences) #remove after testing
    result = sorted(result, key=lambda x: x[1])
    for idx, distance in result[0:BEST_K]:
        long_description = sentences[idx].strip()
        print(long_description, "Company name: {0} (Score: {1})".format(get_company(long_description, "artificial intelligence"), 1-distance))
    end = time.perf_counter()
    print("It took {0} seconds to process {1} many sentences.".format(end - start, len(sentences)))