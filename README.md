### Content Aware RecSys


Your task is to create a system that can recommend movies to users based on the dataset found [here](https://www.kaggle.com/rounakbanik/the-movies-dataset/data). You will need to login to kaggle to download the data. The enriched MovieLens dataset contains the following files:


* movies_metadata.csv: The main Movies Metadata file. Contains information on 45,000 movies featured in the Full MovieLens dataset. Features include posters, backdrops, budget, revenue, release dates, languages, production countries and companies.
* keywords.csv: Contains the movie plot keywords for our MovieLens movies. Available in the form of a stringified JSON Object.
* credits.csv: Consists of Cast and Crew Information for all our movies. Available in the form of a stringified JSON Object.
* links.csv: The file that contains the TMDB and IMDB IDs of all the movies featured in the Full MovieLens dataset.
* links_small.csv: Contains the TMDB and IMDB IDs of a small subset of 9,000 movies of the Full Dataset.
* ratings_small.csv: The subset of 100,000 ratings from 700 users on 9,000 movies.
* ratings.csv: The full ratings dataset.

Besides having the usual user-interaction data, this dataset also has some textual metadata (either in the `metadata.csv` file or in the `keywords.csv` file). Whatever you use as part of your recommendation system, you must use some form of textual data as a feature in your model.


We have defined a helper `recsys.py` file that would help you. It has some boiler plate defined but you need to fill out yourself and write some extra functions or two. You can also solve this using another programming language like Scala , the python code might help you by providing some pseudo-code.

If you use extra libraries, please amend the provided `requirements.txt` file and this readme with instructions. Once everything is ready, we could use the tool by running:

`python recsys.py --in-folder <path-to-data> --out-folder <path-to-model-destination>` , where
	* `<path-to-data>` corresponds to the data containing the csv files
	* `<path-to-model-destination>` corresponds to a folder where the trained model will be serialised to


Your code in `recsys.py` should:
* generate a training / eval / test split (and do any necessary data pre-processing)
* train a model
* print evaluation metrics
* save it to a destination

Once you are done, submit a pull request for evaluation.


---------------------------------------------------------


### AUTHOR
- David Rodrigues


### DATE
- 2021/10/28


### DESCRIPTION
- The below documents key steps I took to produce the recommendations.


### NOTES
- remote repo is: https://github.com/drodri321/optimizely.git
- so if required: 

- git init
- git clone https://github.com/drodri321/optimizely.git
- however I have pulled and attached

- need to pip install lightfm (& numpy, pandas)

- I have a notebook which I used for development before writing up to the python module.  If required saved in 'misc' folder but essentially involved hyperparameter tuning and looks cleaner.  **CAN ALTERNATIVELY RUN SAME CODE AS HERE VIA NOTEBOOK.**

- I have only used 2 of the files in the Kaggle link: small ratings & metadata, and have only downloaded these.

- Furthermore, to minimise space I have created an extracted (much smaller) version of the metadata file by selecting features.

- Done on a MacBook.

- End-to-end execution approx 30-40mins on my PC.

- My PC only has 2 cores.  If this is tested on a PC with more, suggest to up the number of threads in prediction function to match number of cores.  This will speed up processing time, but not essential.


### LIBRARIES
a) recommender.reco_utils
- I extracted from the below which amongst other things provides excellent evaluation metrics specific to recommenders.
- git init
- git clone https://github.com/microsoft/recommenders.git

- Specifically I chose precision + recall, but also has classics like ndcg & map:
- from reco_utils.evaluation.python_evaluation import precision_at_k, recall_at_k

- To keep tidy I only saved the reco_utils subdirectory as there are plenty of others.  As I have already done this, the git commands above are no longer required.

b) lightfm
- Algo of choice - hybrid model which enables item (& user) attributes as inputs.

c) numpy, pandas, sys, os
- Standard imports for mathematical functions, dataframe usage + filepath details.

d) itertools
- To list the films

e) pickle
- To save the model

f) ast.literal_eval
- To extract text from nested lists.


### CLI
- python recsys.py --in-folder r'data/' --out-folder r'model/'


### NEXT STEPS
- Beyond the submission I would explore more item attributes, e.g. budget, production company.
- EDA would be useful to detect anomalies/messy cells.
- Would be good to have data to profile users too, e.g. occupation, age.  LightFM can use these similar to how we used genre.
- Explore other algos, e.g. Deep FM + RBM.
- Aim is to enhance to precision + recall.

---------------------------------------------------------
*END*
