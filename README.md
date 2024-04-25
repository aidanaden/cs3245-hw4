# CS3245 Homework #4

Python version: 3.12.0

### Group:

- A0218327E ()
- A0255343H (e0960013@u.nus.edu)
- A0218135L (e0544171@u.nus.edu)

== General Notes about this assignment ==

### Indexing

#### Dictionary

- Remove numbers, punctuation, stop words (in, and, the, etc)
- Use stem words
- Include n-words without stopword inbetween
  - e.g biword -> “relevance is being modeled” > “being modeled"
- Zones: title, court, date, content
- (word, zone): pointer

#### Posting List

- Calculate TF-normalized for each doc term
- $ doc1 TF1 doc2 TF2 ...
  - '$' indicates start of posting for term
- much of the runtime is taken up by ntlk library functions
  - in the initial implementation, a simple runtime analysis shows that the stemmer takes ~60% of total indexing runtime
  - combine single term and biword indexing into one loop instead of multiple runs for each, reducing number of ntlk library function calls

### Search

#### Query

- convert regular and boolean queries into list of terms
- preliminary search to retrieve relevant docs
- query refinement on relevant docs
- final search on refined query

#### Relevance

- rank td-idf > 0.6 = relevant
- each zone given a weighting that adds up to 1
  - title: 0.2, court: 0.2, date: 0.1, content: 0.5
  - used to calculate document score

#### Query refinement

- Use the given relevant docs (if available) + our own first run relevant docs
- Rocchio / pseudo relevance (choose one)

== Statement of individual work ==

Please put a "x" (without the double quotes) into the bracket of the appropriate statement.

[x] We, A0218327E-A0255343H-A0218135L, certify that I have followed the CS 3245 Information
Retrieval class guidelines for homework assignments. In particular, I/we
expressly vow that I/we have followed the Facebook rule in discussing
with others in doing the assignment and did not take notes (digital or
printed) from the discussions.

[ ] I/We, A0000000X, did not follow the class rules regarding homework
assignment, because of the following reason:

== References ==

nltk (https://www.nltk.org/howto/stem.html, https://www.nltk.org/api/nltk.tokenize.html) -> using nltk tokenize and stemmer
BSBI (https://nlp.stanford.edu/IR-book/html/htmledition/blocked-sort-based-indexing-1.html) -> more info on BSBI
python file management (https://docs.python.org/3/library/os.path.html) -> for managing file directories
python IO (https://docs.python.org/3/tutorial/inputoutput.html) -> python IO methods usage
python pickle (https://docs.python.org/3/library/pickle.html) -> python pickle file storage
rocchio algorithm (https://en.wikipedia.org/wiki/Rocchio_algorithm)
